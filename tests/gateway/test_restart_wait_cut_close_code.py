"""A socket a drain cuts closes 1013 on the wire, so its client can tell a drain cut from a normal end.

Observed on a gateway restart with two kanban cards running: the restart wait cut two kanban submitters' sockets. The gate
hands the handler a ``websocket.disconnect`` with 1013 (``gateway/run_api.py`` ``receive_admitted``,
pinned by ``tests/gateway/test_restart_wait_prompt_reach.py::test_restart_wait_cuts_sockets_opened_before_it``)
and the gateway log said ``client_disconnect(code=1013)``, but the handler's teardown then closed the
socket with its ordinary close: the client read 1000, a normal end. The handler here stands in for
``tui_gateway.ws.handle_ws`` and ends as it does: it reads until the disconnect, then its ``finally``
calls ``ws.close()`` with no code.
"""
import asyncio
import contextlib
import json
import socket
from types import SimpleNamespace

import pytest

FRAME = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "prompt.receipt", "params": {}})


def _owner(tmp_path, monkeypatch):
    """The real gate in front of a stand-in handler, with a runner the test drains."""
    from starlette.websockets import WebSocketDisconnect
    from gateway.run_api import GatewayRuntimeAPI
    from gateway.runtime_bootstrap import TicketStore
    from hermes_cli import web_server as web
    from hermes_cli import web_server_chat

    tickets = TicketStore("owner", [str(tmp_path)])
    authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / "state.db"), profile_id=str(tmp_path), runner=None)
    runner = SimpleNamespace(_draining=False, _running=True, session_runtime_descriptor={"state": "ready"},
                             session_authority=authority, session_ticket_store=tickets)

    async def handle_ws(ws, **kwargs):
        await ws.accept(subprotocol="hermes-gateway-v1")
        try:
            while True:
                request = json.loads(await ws.receive_text())
                await ws.send_text(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"status": "running"}}))
        except WebSocketDisconnect:
            pass
        finally:
            with contextlib.suppress(RuntimeError):  # the server closed it already (its shutdown)
                await ws.close()

    async def fallback(*args):
        pytest.fail("a ticketed native socket must not enter the legacy fallback")

    monkeypatch.setattr("tui_gateway.ws.handle_ws", handle_ws)
    monkeypatch.setattr(web, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(web_server_chat, "_ws_request_is_allowed", lambda ws: True)

    def open_restart_wait():
        runner._draining = True
        runner.session_runtime_descriptor.update(state="draining", drain_reason="restart_wait")

    def ticket():
        return tickets.mint(profile_id=str(tmp_path), subject="human", purpose="interactive")

    return GatewayRuntimeAPI(fallback, runner, web.app), runner, open_restart_wait, ticket


@pytest.mark.asyncio
@pytest.mark.parametrize("opened_in_wait", [False, True])
async def test_the_gate_closes_a_cut_socket_1013_and_leaves_other_closes_alone(tmp_path, monkeypatch, opened_in_wait):
    """The ASGI messages the gate sends the server: a socket opened before the wait is cut, and its
    handler's ordinary close goes out as 1013; one admitted into the wait ends normally, at 1000."""
    gate, _runner, open_restart_wait, ticket = _owner(tmp_path, monkeypatch)
    if opened_in_wait:
        open_restart_wait()
    incoming = iter([{"type": "websocket.connect"}, {"type": "websocket.receive", "text": FRAME},
                     {"type": "websocket.disconnect", "code": 1000}])
    sent = []

    async def receive():
        message = next(incoming)
        if message["type"] == "websocket.receive" and not opened_in_wait:
            open_restart_wait()  # the wait opens while this socket is live: its next frame is cut
        return message

    async def send(message):
        sent.append(message)

    token = ticket()
    scope = {"type": "websocket", "path": "/api/ws", "scheme": "ws", "query_string": b"",
             "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 9999),
             "headers": [(b"sec-websocket-protocol", ("hermes-gateway-v1, hermes-gateway-ticket." + token).encode())],
             "subprotocols": ["hermes-gateway-v1", "hermes-gateway-ticket." + token]}
    await gate(scope, receive, send)
    closes = [message.get("code") for message in sent if message["type"] == "websocket.close"]
    assert closes == [1000 if opened_in_wait else 1013], sent


@contextlib.asynccontextmanager
async def _serving(gate):
    """The gate behind real uvicorn on loopback (lifespan off, no WS ping: the dashboard server's shape)."""
    import uvicorn
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(gate, lifespan="off", log_level="error", timeout_graceful_shutdown=5,
                                           ws_ping_interval=None, ws_ping_timeout=None))
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(500):
            if server.started or task.done():
                break
            await asyncio.sleep(0.01)
        assert server.started, "uvicorn did not start"
        yield server, listener.getsockname()[1]
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


def _dial(port, ticket):
    from websockets.asyncio.client import connect
    return connect(f"ws://127.0.0.1:{port}/api/ws", proxy=None, open_timeout=5,
                   subprotocols=["hermes-gateway-v1", "hermes-gateway-ticket." + ticket])


async def _closed_on_next_frame(client):
    from hermes_cli.gateway_client import GatewayClientError
    with pytest.raises(GatewayClientError) as failed:
        await client.rpc("prompt.receipt", session_id="s", admission_id="a")
    return failed.value


@pytest.mark.asyncio
@pytest.mark.parametrize("cut_by", ["the restart wait opening", "stop() ending the wait"])
async def test_a_cut_reaches_the_client_as_1013(tmp_path, monkeypatch, cut_by):
    """Real uvicorn, Starlette, websockets and GatewayClient: the client reads close 1013 and names it."""
    from hermes_cli.gateway_client import GatewayClient
    gate, runner, open_restart_wait, ticket = _owner(tmp_path, monkeypatch)
    if cut_by == "stop() ending the wait":
        open_restart_wait()  # this socket is admitted INTO the wait; stop() then withdraws it
    async with _serving(gate) as (_server, port), _dial(port, ticket()) as ws, GatewayClient(ws) as client:
        assert await client.rpc("prompt.receipt", session_id="s", admission_id="a") == {"status": "running"}
        if cut_by == "stop() ending the wait":
            runner._running = False
        else:
            open_restart_wait()
        cut = await _closed_on_next_frame(client)
    assert ws.close_code == 1013, f"the drain cut reached the client as close {ws.close_code}"
    assert getattr(cut, "code", None) == 1013 and "drain cut" in str(cut), repr(cut)


@pytest.mark.asyncio
async def test_a_server_shutdown_is_not_named_a_drain_cut(tmp_path, monkeypatch):
    """Control: a close that is not the gate's cut keeps its own code (uvicorn's shutdown: 1012)."""
    from hermes_cli.gateway_client import GatewayClient
    gate, _runner, _open, ticket = _owner(tmp_path, monkeypatch)
    async with _serving(gate) as (server, port), _dial(port, ticket()) as ws, GatewayClient(ws) as client:
        assert await client.rpc("prompt.receipt", session_id="s", admission_id="a") == {"status": "running"}
        server.should_exit = True
        await asyncio.wait_for(client.reader, 10)
        closed = await _closed_on_next_frame(client)
    assert ws.close_code == 1012, ws.close_code
    assert getattr(closed, "code", None) == 1012 and "drain cut" not in str(closed), repr(closed)

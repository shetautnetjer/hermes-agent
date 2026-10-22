"""GatewayClient surfaces the owner's close code. A socket that closes under a request, or between two
requests, fails them with ``GatewayDisconnectedError`` carrying the close code; 1013 is named as a
restart's drain cut.

Observed on a gateway restart with two kanban cards running: a restart's after-turn wait cut the kanban submitters'
sockets, and the cut reached them as an untyped "Gateway disconnected" with the code thrown away.
A close between two requests (stop() closing the socket) escaped as websockets' own
``ConnectionClosed``, past every caller's ``GatewayClientError`` handling. The first two tests run
the real websockets client and server over loopback, with no gateway.
"""
import asyncio
import json

import pytest

from hermes_cli import gateway_client
from hermes_cli.gateway_client import GatewayClient
from tests.waiting import async_wait_until


async def _with_owner(owner, body):
    """Run *body(client)* against a loopback websockets server that runs *owner* per connection."""
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    async with serve(owner, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        async with connect(f"ws://127.0.0.1:{port}", proxy=None, open_timeout=5) as ws:
            async with GatewayClient(ws) as client:
                return await body(client)


@pytest.mark.asyncio
async def test_a_drain_cut_fails_the_pending_request_with_its_close_code():
    async def owner(ws):
        await ws.recv()  # the request a restart's after-turn wait cuts
        await ws.close(1013, "restart_wait")

    async def body(client):
        failures = []
        for _ in range(2):  # the cut request, then one more on the closed socket
            with pytest.raises(gateway_client.GatewayClientError) as failed:
                await client.rpc("prompt.receipt", session_id="s", admission_id="a")
            failures.append(failed.value)
        return failures

    cut, after = await _with_owner(owner, body)
    assert getattr(cut, "code", None) == 1013, f"the close code was thrown away: {cut!r}"
    assert "drain cut" in str(cut) and "turn outcome is unknown" in str(cut), str(cut)
    assert type(after) is type(cut) and getattr(after, "code", None) == 1013, repr(after)


@pytest.mark.asyncio
async def test_a_close_between_requests_fails_the_next_one_with_its_close_code():
    async def owner(ws):
        request = json.loads(await ws.recv())
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": {"status": "running"}}))
        await ws.close(1012, "service restart")  # the server stops between two receipt polls

    async def body(client):
        first = await client.rpc("prompt.receipt", session_id="s", admission_id="a")
        await async_wait_until(lambda: client.reader.done(), what="the owner's close reaching the reader")
        try:
            await client.rpc("prompt.receipt", session_id="s", admission_id="a")
        except Exception as exc:  # health: allow BLE001 -- the test asserts on the exact type below
            return first, exc
        return first, None

    first, closed = await _with_owner(owner, body)
    assert first == {"status": "running"}
    assert isinstance(closed, gateway_client.GatewayClientError), (
        f"a request on a socket the owner closed must fail as a GatewayClientError, not {closed!r}")
    assert getattr(closed, "code", None) == 1012, repr(closed)


@pytest.mark.asyncio
async def test_a_send_on_a_socket_closed_under_it_raises_the_typed_loss():
    """The close has landed but the reader has not run yet: websockets raises from ``send``."""
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    class _Closing:
        def __init__(self):
            self.never = asyncio.Event()

        async def send(self, raw):
            raise ConnectionClosedError(Close(1012, "service restart"), None)

        def __aiter__(self):
            return self

        async def __anext__(self):
            await self.never.wait()

    async with GatewayClient(_Closing()) as client:
        try:
            await client.rpc("prompt.receipt", session_id="s", admission_id="a")
            closed = None
        except Exception as exc:  # health: allow BLE001 -- the test asserts on the exact type below
            closed = exc
    assert isinstance(closed, gateway_client.GatewayClientError), repr(closed)
    assert getattr(closed, "code", None) == 1012, repr(closed)

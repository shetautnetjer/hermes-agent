"""A gateway restart's after-turn wait must not amputate the kanban turn it is waiting for.

Incident (a default-profile gateway): ``Restart requested
with 2 active work unit(s); deferring stop() until they finish (cap=1800s)`` waited on two
``managed`` units, two kanban cards whose dispatcher-spawned submitters
(``hermes_cli/kanban_worker_client.py``) were polling ``prompt.receipt`` on sockets opened while
the owner was ready. About 100 ms later both sockets were closed (1013). Each submitter printed
``Kanban owner refused: Gateway disconnected; turn outcome is unknown`` and exited 1, and
``Managed worker lost: EOFError`` followed twice. At +1.41 s the gateway logged ``Restart deferred
wait complete — active work drained; proceeding to stop()``, and stop() found
``active_at_start=0``.

The chain before the fix:

1. ``gateway/run_api.py`` (``receive_admitted``) cuts a socket opened before the wait on its next
   frame, so that "its client reconnects to the next owner"
   (``tests/gateway/test_restart_wait_prompt_reach.py::test_restart_wait_cuts_sockets_opened_before_it``).
   A re-dial is admitted INTO the wait: ``attachable()`` accepts ``draining``/``restart_wait``, and
   ``prompt.receipt`` is in ``RESTART_WAIT_WS_METHODS``.
2. ``hermes_cli/kanban_worker_client.py`` never re-dials once ``kanban.run`` is accepted. The cut
   reaches it as ``GatewayClient``'s generic "Gateway disconnected; turn outcome is unknown" error,
   not as ``GatewayUnavailableError``, so ``run()`` re-raises it and ``_submit`` exits 1.
3. ``gateway/session_kanban.py::_watch_submitter`` ties the bound managed worker to that process:
   once the submitter's pid is gone, the worker calls ``os._exit(1)`` (EOF: "Managed worker lost").
4. ``gateway/run_shutdown.py::_await_active_work_before_restart`` counts ``not worker.closed``, so
   the amputated turn reads as drained.

The test runs the real links 1, 2 and 4: the gate, ``GatewayClient`` over an in-memory socket, the
submitter's ``run()`` and the restart wait. It models link 3 as written: the turn is lost when its
submitter exits before the turn finishes. It pins the outcome, not a code change: the wait must not
amputate the turn, and the submitter must still report the card's exit code across the restart. A
gate that only keeps pre-wait sockets through the wait does not pass. stop() cuts them right after
the wait, before the submitter reads its terminal receipt, so the card is booked as a crash even
though its turn completed.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from tests.waiting import async_wait_until

SESSION, ADMISSION = "s-card", "a-card"
TURN_POLLS = 3  # receipt polls the card's managed turn still needs once the restart wait opens


class _Socket:
    """One in-memory WebSocket. The client half is the ``websockets`` connection that
    ``GatewayClient`` iterates and sends on. The server half is the ASGI receive/send pair that
    the gate wraps."""

    def __init__(self):
        self.to_server, self.to_client = asyncio.Queue(), asyncio.Queue()

    async def send(self, raw):
        await self.to_server.put({"type": "websocket.receive", "text": raw})

    def __aiter__(self):
        return self

    async def __anext__(self):
        raw = await self.to_client.get()
        if raw is None:  # the server side is gone: the connection closed
            raise StopAsyncIteration
        return raw

    async def receive(self):
        return await self.to_server.get()

    async def asgi_send(self, message):
        if message["type"] == "websocket.send":
            await self.to_client.put(message["text"])
        elif message["type"] == "websocket.close":
            await self.to_client.put(None)


class _Turn:
    """The card's managed turn, as the owner's runtime store and its managed worker see it."""

    def __init__(self):
        self.finished = False
        self.lost: str | None = None  # why the managed worker died before the turn finished
        self.polls_left: int | None = None  # armed when the restart wait opens: receipt polls to come

    @property
    def closed(self):  # ``ManagedWorker.closed``: what the restart wait counts
        return self.finished or self.lost is not None

    def receipt(self):
        status = "terminal" if self.finished else "unknown" if self.lost else "running"
        return {"status": status, "admission_id": ADMISSION}

    def poll(self):
        """One ``prompt.receipt`` answer. Once armed, the turn finishes ``polls_left`` polls later,
        right after answering ``running``. Like a real turn, it ends between two polls, so the
        submitter only learns of it on its next poll."""
        receipt = self.receipt()
        if self.polls_left is not None and receipt["status"] == "running":
            self.polls_left -= 1
            self.finished = self.polls_left <= 0
        return receipt


class _Owner:
    """One gateway instance serving the card's profile: a runner the real gate reads, plus the
    canonical verbs the submitter calls (behind the ``tui_gateway.ws.handle_ws`` seam)."""

    def __init__(self, instance, tmp_path, turn):
        from gateway.runtime_bootstrap import TicketStore

        self.instance, self.profile = instance, str(tmp_path)
        self.tickets = TicketStore(instance, frozenset({self.profile}))
        authority = SimpleNamespace(db=SimpleNamespace(db_path=tmp_path / f"{instance}.db"),
                                    profile_id=self.profile, runner=None)
        self.runner = SimpleNamespace(_draining=False, _running=True, session_runtime_descriptor={"state": "ready"},
                                      session_authority=authority, session_ticket_store=self.tickets)
        self.turn, self.calls, self.cuts = turn, [], []

    def count(self, method):
        return sum(1 for called in self.calls if called == method)

    def answer(self, method):
        from gateway.drain_control import ws_method_refused_while_draining

        self.calls.append(method)
        if ws_method_refused_while_draining(self.runner, method):
            return {"error": {"code": 4009, "message": "runtime_draining"}}
        if method == "kanban.run":
            return {"result": {"session_id": SESSION, "receipt": {"status": "queued", "admission_id": ADMISSION}}}
        if method == "prompt.receipt":
            return {"result": self.turn.poll()}
        return {"error": {"code": 4004, "message": "unknown_method"}}

    def open_restart_wait(self):
        """What ``request_restart()`` does before its after-turn wait: refuse new turns and publish
        the wait, while the owner keeps running its turns."""
        from gateway.drain_control import open_restart_wait

        self.runner._draining = True
        self.runner.session_runtime_descriptor["state"] = "draining"
        open_restart_wait(self.runner)


def _handle_ws(owners, errors):
    async def handle(ws, **kwargs):
        owner = owners.get(kwargs["auth_identity"].get("instance_id"))
        if owner is None:
            errors.append(f"socket bound to an unknown owner: {kwargs['auth_identity']!r}")
            return
        await ws.accept(subprotocol="hermes-gateway-v1")
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                owner.cuts.append(message.get("code"))
                return
            frame = json.loads(message["text"])
            await ws.send_text(json.dumps({"jsonrpc": "2.0", "id": frame["id"], **owner.answer(frame["method"])}))

    return handle


def _connect(current, errors):
    """``connect_gateway()`` as the submitter sees it. It dials the owner that
    ``ensure_gateway_runtime`` hands out (the ready one, or the restart-wait owner, which
    ``attachable()`` accepts) with a fresh interactive ticket, through the real gate."""
    from gateway.run_api import GatewayRuntimeAPI
    from hermes_cli import web_server as web
    from hermes_cli.gateway_client import GatewayClient

    async def legacy_fallback(*_args):
        errors.append("a ticketed native socket entered the legacy fallback")

    def note_server_end(task, sock):
        if not task.cancelled() and task.exception() is not None:
            errors.append(f"gate raised {type(task.exception()).__name__}: {task.exception()}")
        sock.to_client.put_nowait(None)

    @asynccontextmanager
    async def connect(endpoint=None, subject=None):
        owner = current["owner"]
        ticket = owner.tickets.mint(profile_id=owner.profile, subject="human", purpose="interactive")
        protocols = ["hermes-gateway-v1", "hermes-gateway-ticket." + ticket]
        scope = {"type": "websocket", "path": "/api/ws", "scheme": "ws", "query_string": b"",
                 "client": ("127.0.0.1", 1234), "server": ("127.0.0.1", 9999),
                 "headers": [(b"sec-websocket-protocol", ", ".join(protocols).encode())], "subprotocols": protocols}
        sock = _Socket()
        sock.to_server.put_nowait({"type": "websocket.connect"})
        server = asyncio.ensure_future(GatewayRuntimeAPI(legacy_fallback, owner.runner, web.app)(
            scope, sock.receive, sock.asgi_send))
        server.add_done_callback(lambda task: note_server_end(task, sock))
        try:
            async with GatewayClient(sock) as client:
                yield client
        finally:
            sock.to_server.put_nowait({"type": "websocket.disconnect", "code": 1000})
            await asyncio.wait({server}, timeout=2)

    return connect


def _wire(monkeypatch, owners, current):
    from gateway import session_kanban
    from hermes_cli import gateway_client, web_server as web, web_server_chat

    errors = []
    monkeypatch.setattr("tui_gateway.ws.handle_ws", _handle_ws(owners, errors))
    monkeypatch.setattr(web, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True)
    monkeypatch.setattr(web_server_chat, "_ws_request_is_allowed", lambda ws: True)
    monkeypatch.setattr(gateway_client, "connect_gateway", _connect(current, errors))
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    return errors


def _restart_waiter(turn) -> Any:
    """The runner surface ``_await_active_work_before_restart`` reads: one managed unit, counted
    while its worker is not closed (``uncounted_runtime_work``). It stands in for the mixin's
    ``self``, hence ``Any``."""
    return SimpleNamespace(
        _active_work_count=lambda: 0 if turn.closed else 1,
        _awaitable_work_count=lambda: 0 if turn.closed else 1,
        _wedged_agent_count=lambda: 0, _restart_safe_cron_count=lambda: 0,
        _describe_active_work=lambda: [] if turn.closed else [{"kind": "managed", "pid": 0}],
        _scale_to_zero_status=lambda *_: None, _restart_after_turn_timeout=30.0)


def _outcome(task):
    if not task.done():
        return "still running"
    if task.cancelled():
        return "cancelled"
    if task.exception() is not None:
        return f"raised {type(task.exception()).__name__}: {task.exception()}"
    return f"returned {task.result()!r}"


async def _watch_submitter(submitter, turn):
    """``gateway/session_kanban.py::_watch_submitter`` as written: the bound managed worker checks
    its submitter's liveness every second and calls ``os._exit(1)`` once that process is gone. A
    submitter that exits before the turn finishes therefore takes the turn with it."""
    await asyncio.wait({submitter})
    if not turn.finished:
        turn.lost = f"its submitter exited first ({_outcome(submitter)})"


async def _start_card(tmp_path):
    from hermes_cli import kanban_worker_client

    return asyncio.ensure_future(kanban_worker_client.run({"task_id": "t-card"}, str(tmp_path / "kanban.db")))


@pytest.mark.asyncio
async def test_restart_wait_does_not_amputate_the_kanban_turn_it_waits_for(tmp_path, monkeypatch):
    from gateway.run_shutdown import GatewayShutdownMixin

    turn = _Turn()
    old, new = _Owner("owner-before", tmp_path, turn), _Owner("owner-after", tmp_path, turn)
    current = {"owner": old}
    errors = _wire(monkeypatch, {old.instance: old, new.instance: new}, current)
    loop = asyncio.get_running_loop()

    submitter = await _start_card(tmp_path)
    watcher = asyncio.ensure_future(_watch_submitter(submitter, turn))
    # Mid-turn: the submitter polls its receipt on a socket it opened while the owner was ready.
    await async_wait_until(lambda: old.count("prompt.receipt") >= 2 or submitter.done(), timeout=10,
                           what="the card polling its receipt")
    assert not submitter.done(), f"harness: the card ended before the restart: {_outcome(submitter)}; {errors}"

    old.open_restart_wait()
    turn.polls_left = TURN_POLLS
    opened = loop.time()
    drained = await asyncio.wait_for(
        GatewayShutdownMixin._await_active_work_before_restart(_restart_waiter(turn)), 15)
    waited = loop.time() - opened
    # stop(): the wait is over. The old owner's sockets are cut, and a re-dial reaches the next owner.
    old.runner._running = False
    current["owner"] = new
    await asyncio.wait({submitter}, timeout=15)
    await asyncio.wait({watcher}, timeout=5)

    assert not errors, errors
    assert turn.finished and turn.lost is None, (
        f"the restart wait reported {'drained' if drained else 'not drained'} {waited:.2f}s after it opened, "
        f"but the managed turn it was waiting for (it still needed {TURN_POLLS} receipt polls) never finished: "
        f"it was lost because {turn.lost}. The old owner closed the card's sockets with {old.cuts}, and the "
        f"submitter made no further call after the cut (old owner calls: {old.calls}; next owner calls: "
        f"{new.calls}). This is the reported incident: 'Restart deferred wait complete — active "
        "work drained' at +1.41 s, after two 'Managed worker lost: EOFError'.")
    assert _outcome(submitter) == "returned 0", (
        "the card's submitter must report its exit code across the restart; a cut after admission must "
        f"not end it (stop() cuts every socket right after the wait): {_outcome(submitter)}")
    assert old.count("kanban.run") + new.count("kanban.run") == 1, "the claim must be submitted exactly once"


@pytest.mark.asyncio
async def test_harness_without_a_restart_the_card_returns_its_exit_code(tmp_path, monkeypatch):
    """Control for the harness: with no restart the real submitter, the real ``GatewayClient`` and the
    real gate carry the card to its terminal receipt, with one submission and no 1013 cut."""
    turn = _Turn()
    owner = _Owner("owner-before", tmp_path, turn)
    errors = _wire(monkeypatch, {owner.instance: owner}, {"owner": owner})
    submitter = await _start_card(tmp_path)
    await async_wait_until(lambda: owner.count("prompt.receipt") >= 2 or submitter.done(), timeout=10,
                           what="the card polling its receipt")
    turn.finished = True
    await asyncio.wait({submitter}, timeout=10)
    assert not errors, errors
    assert _outcome(submitter) == "returned 0"
    assert owner.count("kanban.run") == 1
    assert 1013 not in owner.cuts, owner.cuts


@pytest.mark.asyncio
async def test_harness_a_redial_inside_the_wait_is_admitted_and_answered(tmp_path, monkeypatch):
    """Control for the premise: during the restart wait, the base gateway admits a fresh dial and
    answers ``prompt.receipt`` on it, while it refuses a second ``kanban.run``. The cut submitter
    could resume this way without submitting its claim twice."""
    from hermes_cli import gateway_client

    turn = _Turn()
    owner = _Owner("owner-before", tmp_path, turn)
    errors = _wire(monkeypatch, {owner.instance: owner}, {"owner": owner})
    owner.open_restart_wait()
    async with gateway_client.connect_gateway() as client:
        receipt = await client.rpc("prompt.receipt", session_id=SESSION, admission_id=ADMISSION)
        with pytest.raises(gateway_client.GatewayClientError, match="runtime_draining"):
            await client.rpc("kanban.run", task_id="t-card")
    assert not errors, errors
    assert receipt == {"status": "running", "admission_id": ADMISSION}
    assert 1013 not in owner.cuts, owner.cuts

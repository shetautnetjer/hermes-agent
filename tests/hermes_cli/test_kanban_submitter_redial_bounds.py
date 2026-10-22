"""After ``kanban.run`` is accepted, a lost owner connection is re-dialed and the same admission's
receipt resumed, never resubmitted, within bounds that cover the owner's own restart.

Observed on a gateway restart with two kanban cards running: a graceful restart's after-turn wait cut two cards'
submitter sockets (1013); each submitter exited 1 and its managed turn died with it
(``tests/hermes_cli/test_kanban_submitter_survives_restart_wait.py``). Re-dialing fixes that; these
pin the bounds the re-dial needs:

(a) once admitted, an owner that says it is draining or starting is waited for as long as its restart
    may hold the turn (``agent.restart_after_turn_timeout``, then stop() and the next owner's start),
    counted from when it stopped answering, not ``OWNER_WAIT_S`` from the submitter's start;
(b) re-dials that reach no owner ("Gateway connection failed", a control socket gone mid-dial, a
    refused upgrade) are bounded, so a dead host is reported rather than spun on;
(c) the close code reaches the log: a 1013 drain cut is named as one.

Time is simulated: the submitter's deadline clock and pauses are substituted, nothing sleeps.
"""
import asyncio
from contextlib import asynccontextmanager

import pytest

from hermes_cli import gateway_client, kanban_worker_client

ADMISSION = "a1"


def _hang_up(code=None):
    """What GatewayClient raises once the owner closes the socket (before the fix: the untyped error)."""
    typed = getattr(gateway_client, "GatewayDisconnectedError", None)
    if typed is None:
        return gateway_client.GatewayClientError("Gateway disconnected; turn outcome is unknown.")
    return typed(code)


def _unavailable(state):
    return gateway_client.GatewayUnavailableError(f"Gateway {state}: not_ready", state)


class _Clock:
    """The submitter's deadline clock: advanced by its pauses and by the owner's own delays."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    async def pause(self, seconds):
        self.now += seconds
        await asyncio.sleep(0)


class _Owner:
    """``connect_gateway`` as the submitter sees it. ``script(dial, now)`` decides each dial: an
    exception is the dial's own failure; a list holds the answers that connection gives, one per
    request (a receipt status, or an exception the request raises, such as a hang-up)."""

    def __init__(self, clock, script, *, request_cost=0.0, refusal_cost=0.0):
        self.clock, self.script = clock, script
        self.request_cost, self.refusal_cost = request_cost, refusal_cost
        self.calls, self.dials = [], 0
        self.cut_at = None

    def connect(self):
        owner = self

        @asynccontextmanager
        async def connect(endpoint=None, subject=None):
            owner.dials += 1
            step = owner.script(owner.dials, owner.clock.now)
            if isinstance(step, BaseException):
                owner.clock.now += owner.refusal_cost
                raise step
            answers = list(step)

            class _Client:
                events: asyncio.Queue = asyncio.Queue()

                async def rpc(self, method, **params):
                    owner.calls.append(method)
                    owner.clock.now += owner.request_cost
                    answer = answers.pop(0)
                    if isinstance(answer, BaseException):
                        owner.cut_at = owner.clock.now if owner.cut_at is None else owner.cut_at
                        raise answer
                    receipt = {"status": answer, "admission_id": ADMISSION}
                    return {"session_id": "s1", "receipt": receipt} if method == "kanban.run" else receipt

            yield _Client()

        return connect


@pytest.fixture
def card(monkeypatch, tmp_path):
    """Start the real submitter's ``run()`` against a scripted owner, on the simulated clock."""
    clock = _Clock()
    monkeypatch.setattr(kanban_worker_client, "_monotonic", clock, raising=False)
    monkeypatch.setattr(kanban_worker_client, "_pause", clock.pause, raising=False)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    def start(owner):
        monkeypatch.setattr(gateway_client, "connect_gateway", owner.connect())
        return asyncio.ensure_future(kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")))

    return clock, start


async def _outcome(task):
    """The submitter's end, as text: its return value, or the error it raised."""
    await asyncio.wait({task}, timeout=10)  # a real-time guard only: the scripts never wait
    if not task.done():
        task.cancel()
        return "still running"
    if task.exception() is not None:
        return f"raised {type(task.exception()).__name__}: {task.exception()}"
    return f"returned {task.result()!r}"


@pytest.mark.asyncio
async def test_an_admitted_card_waits_out_its_owners_restart_past_owner_wait_s(card, capsys):
    """(a) + (c): the card ran 1000 s, well past ``OWNER_WAIT_S`` from the submitter's start; then the
    restart's drain cut (1013) and an owner that is draining, then starting, for 1500 s (stop() and
    the next owner's start, inside its restart envelope). The submitter keeps re-dialing, never
    resubmits, and returns the card's exit code from the next owner's receipt."""
    clock, start = card
    unavailable_for = 1500.0

    def script(dial, now):
        if dial == 1:  # admitted, polled for ~1000 s, then the restart wait's drain cut
            return ["queued"] + ["running"] * 10 + [_hang_up(1013)]
        cut = owner.cut_at or 0.0
        if now < cut + unavailable_for:
            return _unavailable("draining" if now < cut + 1000 else "starting")
        return ["running", "terminal"]

    owner = _Owner(clock, script, request_cost=100.0, refusal_cost=30.0)
    outcome = await _outcome(start(owner))
    assert owner.cut_at is not None and owner.cut_at > kanban_worker_client.OWNER_WAIT_S
    assert unavailable_for > kanban_worker_client.OWNER_WAIT_S
    assert outcome == "returned 0", (
        f"the card's owner went away {owner.cut_at:.0f}s after the submitter started and was back "
        f"{unavailable_for:.0f}s later, inside its restart envelope; the submitter must wait for it and "
        f"resume the receipt, not end the card: {outcome} after {owner.dials} dial(s)")
    assert owner.calls.count("kanban.run") == 1, owner.calls
    err = capsys.readouterr().err
    assert "close 1013: drain cut" in err, f"the drain cut's close code must reach the log: {err!r}"
    assert "never resubmitting" in err, err


def test_the_resume_budget_covers_the_owners_restart_after_turn_timeout(tmp_path, monkeypatch):
    """(a): the bound is read the way the owner reads it, from every home whose gateway may own the
    profile. A named profile is served by its root's multiplexer, so the root's 3600 s after-turn
    wait counts even though the profile's own config says 1800 s; an env override counts as well."""
    from gateway.restart import (
        DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT, DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
        DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT, resolve_restart_exit_wait_budget,
    )
    for var in ("HERMES_RESTART_AFTER_TURN_TIMEOUT", "HERMES_RESTART_DRAIN_TIMEOUT", "HERMES_CRON_DRAIN_TIMEOUT"):
        monkeypatch.delenv(var, raising=False)
    budget = getattr(kanban_worker_client, "resume_wait_budget", lambda: kanban_worker_client.OWNER_WAIT_S)

    def envelope(after_turn):
        return resolve_restart_exit_wait_budget(
            DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT, after_turn, DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT)

    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(bare))
    assert budget() >= envelope(DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT) > DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT, (
        f"an admitted card's wait for its owner ({budget():.0f}s) must cover the default after-turn wait "
        f"({DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT:.0f}s) plus stop(), not OWNER_WAIT_S")

    root = tmp_path / "host"
    profile = root / "profiles" / "worker"
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("agent:\n  restart_after_turn_timeout: 3600\n")
    (profile / "config.yaml").write_text("agent:\n  restart_after_turn_timeout: 1800\n")
    monkeypatch.setenv("HERMES_HOME", str(profile))
    assert budget() >= envelope(3600), f"the multiplexer root's 3600 s after-turn wait is not covered: {budget():.0f}s"

    monkeypatch.setenv("HERMES_RESTART_AFTER_TURN_TIMEOUT", "7200")
    assert budget() >= envelope(7200), f"the env override is not covered: {budget():.0f}s"


@pytest.mark.asyncio
async def test_an_admitted_card_gives_up_once_its_owner_stays_unavailable_past_the_budget(card, monkeypatch):
    """(a)'s other side: an owner that keeps saying draining is waited for only as long as the budget,
    then the submitter reports the outcome unknown (it never resubmits) instead of waiting forever."""
    clock, start = card
    monkeypatch.setattr(kanban_worker_client, "resume_wait_budget", lambda: 2000.0, raising=False)

    def script(dial, now):
        if dial == 1:
            return ["queued", "running", _hang_up(1013)]
        return _unavailable("draining")

    owner = _Owner(clock, script, request_cost=1.0, refusal_cost=50.0)
    outcome = await _outcome(start(owner))
    assert "outcome is unknown" in outcome and "not resubmitted" in outcome, outcome
    assert owner.cut_at is not None and clock.now - owner.cut_at >= 2000.0, (
        f"gave up {clock.now - (owner.cut_at or 0):.0f}s after the owner stopped answering, before its 2000 s budget: {outcome}")
    assert owner.dials <= 2000.0 / 50.0 + 3, f"{owner.dials} dials for a 2000 s wait"
    assert owner.calls.count("kanban.run") == 1, owner.calls


def _dial_failure(kind):
    if kind == "connection_failed":  # connect_gateway: discovery said ready, the socket refused
        return gateway_client.GatewayClientError("Gateway connection failed; no local fallback")
    if kind == "control_socket_gone":  # the ticket request found the control socket gone
        return ConnectionRefusedError(111, "Connection refused")
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatus
    from websockets.http11 import Response
    return InvalidStatus(Response(403, "Forbidden", Headers()))  # the gate refused the upgrade


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["connection_failed", "control_socket_gone", "upgrade_refused"])
async def test_redials_that_reach_no_owner_are_bounded(card, kind):
    """(b): after the hang-up every dial fails outright. The submitter re-dials a bounded number of
    times, then reports the outcome unknown; it never spins and never resubmits."""
    clock, start = card
    bound = getattr(kanban_worker_client, "REDIAL_FAILURES", 5)

    def script(dial, now):
        if dial == 1:
            return ["queued", "running", _hang_up(1006)]
        if dial > bound + 50:
            raise AssertionError("the submitter is spinning on a dead host")
        return _dial_failure(kind)

    owner = _Owner(clock, script, request_cost=1.0, refusal_cost=1.0)
    outcome = await _outcome(start(owner))
    assert owner.dials == 1 + bound, f"{owner.dials} dials; wanted the first plus {bound} re-dials: {outcome}"
    assert "re-dials in a row reached no owner" in outcome and "not resubmitted" in outcome, outcome
    assert owner.calls.count("kanban.run") == 1, owner.calls


@pytest.mark.asyncio
async def test_a_redial_that_reaches_the_owner_resets_the_bound(card):
    """(b): the bound counts failed re-dials in a row. One that reaches the owner and reads the
    receipt starts the count again, so two short outages in one turn do not add up to a give-up."""
    clock, start = card
    bound = getattr(kanban_worker_client, "REDIAL_FAILURES", 5)
    refused = gateway_client.GatewayClientError("Gateway connection failed; no local fallback")
    outage = bound - 1
    plan = ([["queued", "running", _hang_up(1013)]] + [refused] * outage + [["running", _hang_up(1012)]]
            + [refused] * outage + [["running", "terminal"]])

    def script(dial, now):
        return plan[dial - 1] if dial <= len(plan) else AssertionError("dialed past the plan")

    owner = _Owner(clock, script, request_cost=1.0, refusal_cost=1.0)
    outcome = await _outcome(start(owner))
    assert outcome == "returned 0", f"{outcome} after {owner.dials} dials"
    assert owner.dials == len(plan) and owner.calls.count("kanban.run") == 1, (owner.dials, owner.calls)

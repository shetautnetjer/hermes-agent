"""After ``kanban.run`` is accepted, re-dials attach only to an owner that is up: they never start one.

The submitter's first dial ensures an owner, starting one if none runs, as every client bootstrap
does. Once its claim is admitted, a re-dial goes through ``connect_gateway(endpoint=RUNNING_OWNER)``,
which dials the owner discovery finds up (ready, or in a restart's after-turn wait) and never starts
one. Otherwise the submitters of a gateway stopped on purpose would start it again, and the card
client's own process tree would become an owner (``tests/gateway/test_kanban_recovery.py`` traces that
tree for canonical-state writes). An owner that is not up (``absent``) is a miss: misses back off and
end the wait after a bound that spans a supervisor's respawn gap (launchd's ThrottleInterval, 30 s).

Time is simulated as in ``test_kanban_submitter_redial_bounds.py``, whose scripted owner this reuses.
"""
import asyncio

import pytest

from hermes_cli import gateway_client, kanban_worker_client
from tests.hermes_cli.test_kanban_submitter_redial_bounds import _Clock, _hang_up, _Owner, _outcome, _unavailable

_NOT_YET = object()  # stands in for RUNNING_OWNER on a tree without it


@pytest.fixture
def submitter(monkeypatch, tmp_path):
    """Start the real submitter's ``run()`` against a scripted owner, on the simulated clock (the
    ``card`` fixture of ``test_kanban_submitter_redial_bounds.py``)."""
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


def _running_owner():
    return getattr(gateway_client, "RUNNING_OWNER", _NOT_YET)


def _recorded(monkeypatch, owner):
    """Patch ``connect_gateway`` with *owner*'s, recording the ``endpoint`` each dial asked for."""
    inner, endpoints = owner.connect(), []

    def connect(endpoint=None, subject=None):
        endpoints.append(endpoint)
        return inner(endpoint, subject)

    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    return endpoints


@pytest.mark.asyncio
async def test_a_redial_after_admission_attaches_only_to_an_owner_that_is_up(submitter, monkeypatch):
    clock, start = submitter

    def script(dial, now):
        if dial == 1:
            return ["queued", "running", _hang_up(1013)]
        if dial in (2, 3):  # the old owner has exited; its replacement has not taken its lock yet
            return _unavailable("absent")
        return ["running", "terminal"]

    owner = _Owner(clock, script, request_cost=1.0, refusal_cost=0.1)
    task = start(owner)
    endpoints = _recorded(monkeypatch, owner)
    outcome = await _outcome(task)
    assert outcome == "returned 0", f"{outcome} after {owner.dials} dial(s)"
    assert endpoints[0] is None, f"the first dial ensures an owner as before, not {endpoints[0]!r}"
    assert endpoints[1:] and all(endpoint is _running_owner() for endpoint in endpoints[1:]) \
        and _running_owner() is not _NOT_YET, (
        f"every re-dial after admission must attach only (RUNNING_OWNER), never start an owner: {endpoints!r}")
    assert owner.calls.count("kanban.run") == 1, owner.calls


@pytest.mark.asyncio
async def test_an_owner_back_within_a_supervisors_respawn_gap_is_resumed(submitter):
    """launchd respawns a KeepAlive job at most every 30 s (ThrottleInterval); until the replacement
    takes its runtime lock, discovery reads ``absent``. A submitter must still be dialing then."""
    clock, start = submitter
    gap = 30.0

    def script(dial, now):
        if dial == 1:
            return ["queued", "running", _hang_up(1013)]
        if now < (owner.cut_at or 0.0) + gap:
            return _unavailable("absent")
        return ["running", "terminal"]

    owner = _Owner(clock, script, request_cost=1.0, refusal_cost=0.1)
    outcome = await _outcome(start(owner))
    assert outcome == "returned 0", (
        f"the owner was back {gap:.0f}s after it went away; the submitter must still be re-dialing: "
        f"{outcome} after {owner.dials} dial(s), {clock.now - (owner.cut_at or 0.0):.0f}s")
    assert owner.calls.count("kanban.run") == 1, owner.calls


@pytest.mark.asyncio
async def test_an_owner_that_never_comes_back_is_reported_within_a_minute(submitter):
    """A gateway stopped on purpose (or a dead host) stays away: the submitter reports the outcome
    unknown once the bound is spent, past the respawn gap and well inside a minute, never resubmitting."""
    clock, start = submitter

    def script(dial, now):
        if dial == 1:
            return ["queued", "running", _hang_up(1012)]
        if dial > 50:
            raise AssertionError("the submitter is spinning on an owner that is gone")
        return _unavailable("absent")

    owner = _Owner(clock, script, request_cost=1.0, refusal_cost=0.1)
    outcome = await _outcome(start(owner))
    waited = clock.now - (owner.cut_at or 0.0)
    assert "outcome is unknown" in outcome and "not resubmitted" in outcome, outcome
    assert 30.0 <= waited < 60.0, f"gave up {waited:.0f}s after the owner went away: {outcome}"
    assert owner.calls.count("kanban.run") == 1, owner.calls


@pytest.mark.asyncio
async def test_the_running_owner_dial_never_starts_a_gateway(tmp_path, monkeypatch):
    """The real ``connect_gateway(endpoint=RUNNING_OWNER)`` on a home with no owner: it reports the
    owner absent and never ensures, starts or spawns one."""
    from hermes_cli import gateway_runtime, gateway_runtime_start
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)
    started = []
    monkeypatch.setattr(gateway_runtime, "ensure_gateway_runtime", lambda *a, **k: started.append("ensure"))
    monkeypatch.setattr(gateway_runtime_start, "spawn_unmanaged_gateway", lambda *a, **k: started.append("spawn"))
    with pytest.raises(gateway_client.GatewayClientError) as refused:
        async with gateway_client.connect_gateway(endpoint=_running_owner()):
            pytest.fail("dialed an owner that does not exist")
    assert isinstance(refused.value, gateway_client.GatewayUnavailableError), repr(refused.value)
    assert refused.value.state == "absent", refused.value.state
    assert started == [], started

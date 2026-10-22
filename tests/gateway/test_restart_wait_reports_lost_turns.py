"""A restart's after-turn wait reports a managed turn whose worker was lost as LOST, not drained.

Observed on a gateway restart with two kanban cards running: "Restart requested
with 2 active work unit(s); deferring stop()", then two "Managed worker lost: EOFError" (the cards'
submitters had exited, so ``_watch_submitter`` ended each worker), then "Restart deferred wait
complete — active work drained" at +1.41 s. The wait counts managed units as ``not worker.closed``,
so a lost turn and a finished one looked the same. A lost turn's outcome is unknown; the log must say
lost, and the wait must not report it drained.
"""
import asyncio
import logging
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.gateway.restart_test_helpers import make_restart_runner
from tests.waiting import async_wait_until


def _worker():
    from gateway.session_managed_worker import ManagedWorker
    return ManagedWorker(SimpleNamespace(pid=0, poll=lambda: 0))


def _waiter(authority):
    runner, _adapter = make_restart_runner()
    runner._restart_after_turn_timeout = 30.0
    runner.session_authority = authority
    return runner


@pytest.mark.asyncio
async def test_a_turn_lost_during_the_wait_is_logged_lost_not_drained(caplog):
    caplog.set_level(logging.INFO, logger="gateway.run")
    finished, lost = _worker(), _worker()
    runner = _waiter(SimpleNamespace(_managed_workers={"s-done": finished, "s-lost": lost}))
    wait = asyncio.ensure_future(runner._await_active_work_before_restart())
    await async_wait_until(lambda: "deferring stop()" in caplog.text or wait.done(), what="the wait opening")
    finished.closed.set()  # its turn completed
    lost.lost = "EOFError"  # what execute_managed records when the worker dies mid-turn
    lost.closed.set()
    drained = await asyncio.wait_for(wait, 10)
    assert "LOST, not drained" in caplog.text and "EOFError" in caplog.text, caplog.text
    assert "active work drained" not in caplog.text, caplog.text
    assert drained is False


@pytest.mark.asyncio
async def test_turns_that_finish_during_the_wait_still_read_drained(caplog):
    """Control: finished turns are drained, and nothing is reported lost."""
    caplog.set_level(logging.INFO, logger="gateway.run")
    first, second = _worker(), _worker()
    runner = _waiter(SimpleNamespace(_managed_workers={"s1": first, "s2": second}))
    wait = asyncio.ensure_future(runner._await_active_work_before_restart())
    await async_wait_until(lambda: "deferring stop()" in caplog.text or wait.done(), what="the wait opening")
    first.closed.set()
    second.closed.set()
    assert await asyncio.wait_for(wait, 10) is True
    assert "active work drained" in caplog.text and "LOST" not in caplog.text, caplog.text


CHILD = """import json, os, sys, psutil
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
out.write((json.dumps({'type': 'hello', 'pid': me.pid, 'birth': me.create_time(),
                       'ancestors': [p.pid for p in me.parents()[:3]]}) + '\\n').encode())
for line in sys.stdin.buffer:  # its bootstrap: the turn has started
    break
open(sys.argv[1]).read()  # held mid-turn until the test lets it go
os._exit(1)  # gone before its turn finished, as when its submitter exits first
"""


@pytest.mark.platforms("linux")
def test_a_managed_worker_lost_mid_wait_is_reported_lost(tmp_path, monkeypatch, caplog):
    """The real ``execute_managed`` loss path and the real wait: a managed turn starts, the restart
    wait opens on it, then its worker dies before the turn finishes."""
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Principal, SessionRef, Submission
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input

    caplog.set_level(logging.INFO, logger="gateway.run")
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    owner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                            _adapter_for_source=lambda source: None)
    authority = SessionAuthority(owner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    held = tmp_path / 'held'
    os.mkfifo(held)

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD, str(held)], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)
    waiter = _waiter(authority)

    def started():
        return any(worker.writer.is_alive() for worker in getattr(authority, '_managed_workers', {}).values())

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        turn = asyncio.ensure_future(managed.execute_managed(authority, ref, row, policy))
        await async_wait_until(lambda: started() or turn.done(), timeout=30, what="the managed turn starting")
        assert not turn.done(), turn
        wait = asyncio.ensure_future(waiter._await_active_work_before_restart())
        await async_wait_until(lambda: "deferring stop()" in caplog.text or wait.done(), what="the wait opening")
        await asyncio.to_thread(lambda: open(held, 'w').close())  # the worker dies mid-turn
        with pytest.raises(managed.ManagedExecutionUnknown):
            await asyncio.wait_for(turn, 30)
        return await asyncio.wait_for(wait, 30)

    with db:
        drained = asyncio.run(scenario())
    assert "Managed worker lost" in caplog.text, caplog.text
    assert "LOST, not drained" in caplog.text, caplog.text
    assert "active work drained" not in caplog.text, caplog.text
    assert drained is False

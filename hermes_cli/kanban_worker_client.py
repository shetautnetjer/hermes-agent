"""Dispatcher subprocess: submit one claim to its profile owner; never run an agent."""
# Spawned as a bare ``sys.executable -m``: in a PM install that interpreter carries no dependencies
# until hermes_bootstrap selects the committed environment, so it must be the first import.
try:
    import hermes_bootstrap
except ModuleNotFoundError as exc:
    if exc.name != "hermes_bootstrap":
        raise

import asyncio
import os
from pathlib import Path
import sys
import time

# A worker the dispatcher spawns while `hermes update` drains the owner (or while the new owner is
# still starting) waits for it instead of exiting: the condition is the host's, not the card's, and
# a failed exit here spends the card's failure budget on an update window. Nothing is submitted
# until an owner is ready, so waiting can never run the claim twice.
OWNER_WAIT_S = 180.0
_TRANSIENT_OWNER_STATES = frozenset({'draining', 'starting'})
# Past the board lock wait: the owner's session setup and receipt before and after the write.
ADMISSION_ANSWER_MARGIN_S = 60.0
# Once ``kanban.run`` has answered, the claim is the owner's and is never submitted again; losing the
# socket after that is a transport event, not the card's outcome. A graceful restart's after-turn wait
# cuts every socket opened before it (close 1013) so that its clients re-dial: the waiting owner, and
# the next owner after it, still answer this admission's ``prompt.receipt``. This process is also the
# card's handle (the bound managed worker ends its turn once it exits: ``_watch_submitter``), so a
# hang-up re-dials and resumes the receipt instead of exiting. A re-dial attaches only to an owner that
# is up (``RUNNING_OWNER``), never starts one: a gateway stopped on purpose stays stopped. An owner that
# says it is draining or starting is waited for up to ``resume_wait_budget()`` without a receipt
# answer. Re-dials that reach no owner (none up, "Gateway connection failed", a refused upgrade, a
# hang-up before any answer) back off 1, 2, 4, 8 and 16 s, and the wait ends at the
# ``REDIAL_FAILURES``-th in a row: 31 s spans the longest gap a supervisor leaves before the next owner
# takes its runtime lock and reads ``starting`` (launchd's ThrottleInterval, 30 s; systemd's
# RestartSec, 5 s), so a dead host is reported rather than spun on.
REDIAL_FAILURES = 6
REDIAL_PAUSE_S = 1.0
RECEIPT_POLL_S = 0.2
_SETTLED = frozenset({'terminal', 'unknown'})
_monotonic = time.monotonic  # the wait deadlines' clock (never the event loop's own)


def admission_answer_budget():
    """Seconds to wait for ``kanban.run``'s answer: the longest the owner's admission write can
    wait on the board lock (``write_txn`` retries ``BEGIN IMMEDIATE`` ``_BUSY_MAX_RETRIES`` times,
    each under the busy timeout) plus ``ADMISSION_ANSWER_MARGIN_S``."""
    from hermes_cli import kanban_db_connect
    lock_wait = (kanban_db_connect._BUSY_MAX_RETRIES + 1) * kanban_db_connect._resolve_busy_timeout_ms() / 1000
    return lock_wait + ADMISSION_ANSWER_MARGIN_S


def resume_wait_budget():
    """Seconds an admitted card's submitter keeps re-dialing an owner that says it is draining or
    starting, counted from the owner's last receipt answer: that owner's own restart envelope
    (``resolve_restart_exit_wait_budget``, which ``hermes gateway restart`` waits out too: the
    after-turn wait, ``agent.restart_after_turn_timeout``, that may be holding this very turn, then
    stop()'s drain and headroom) plus the next owner's start (``DEFAULT_ENSURE_TIMEOUT``), and never
    less than ``OWNER_WAIT_S``.

    Read as the owner reads it (env first, then ``agent.*`` in its effective config) for each home
    whose gateway may own this profile: its own, and the root whose multiplexer serves a named
    profile. The longest wins: outlasting the owner only delays reporting one that never came back,
    while ending first amputates a turn the restart was still waiting for."""
    from hermes_cli.gateway_runtime import DEFAULT_ENSURE_TIMEOUT, gateway_profile_home
    from hermes_cli.gateway_runtime_multiplex import multiplexer_root_for
    from hermes_constants import get_hermes_home
    home = gateway_profile_home(get_hermes_home())
    owners = [home] + [root for root in (multiplexer_root_for(home),) if root is not None]
    return max(OWNER_WAIT_S, max(_restart_envelope(owner) for owner in owners) + DEFAULT_ENSURE_TIMEOUT)


def _restart_envelope(home):
    from gateway.restart import (
        parse_cron_drain_timeout, parse_restart_after_turn_timeout, parse_restart_drain_timeout,
        resolve_restart_exit_wait_budget,
    )
    from hermes_cli.config_effective import load_user_config_effective
    try:
        config = load_user_config_effective(Path(home) / 'config.yaml')
    except Exception:  # health: allow BLE001 -- fail open like the owner's own loader: its defaults apply
        config = {}
    agent = config.get('agent') if isinstance(config, dict) else None
    agent = agent if isinstance(agent, dict) else {}

    def setting(env_var, key, parse):
        raw = os.environ.get(env_var, '')
        return parse(raw if raw.strip() else agent.get(key))
    return resolve_restart_exit_wait_budget(
        setting('HERMES_RESTART_DRAIN_TIMEOUT', 'restart_drain_timeout', parse_restart_drain_timeout),
        setting('HERMES_RESTART_AFTER_TURN_TIMEOUT', 'restart_after_turn_timeout', parse_restart_after_turn_timeout),
        setting('HERMES_CRON_DRAIN_TIMEOUT', 'cron_drain_timeout', parse_cron_drain_timeout))


async def _pause(seconds):
    await asyncio.sleep(seconds)


def _cause(exc):
    """Why the owner did not answer, in a few words (its close code when it closed the socket)."""
    if isinstance(exc, TimeoutError):
        return 'no receipt answer in time'
    return str(exc).split(';')[0] or type(exc).__name__


class _Claim:
    """The card's admission once ``kanban.run`` answered, and the bounds on re-dialing for it: the
    owner may go without a receipt answer for ``budget`` seconds while it says it is draining or
    starting, and at most ``REDIAL_FAILURES`` re-dials in a row may end without reaching it."""

    def __init__(self):
        self.session_id = self.receipt = self.admission_id = None
        self.budget = OWNER_WAIT_S
        self.silent_since = None  # the first loss since the owner last answered
        self.failures = 0
        self.answered_here = False  # this connection has answered (a hang-up then is no failed re-dial)
        self.waiting = None  # the transient state last reported, so a long wait is one log line

    @property
    def admitted(self):
        return self.receipt is not None

    def admit(self, session_id, receipt, budget):
        self.session_id, self.admission_id, self.budget = session_id, receipt['admission_id'], budget
        self.answered(receipt)

    def answered(self, receipt):
        self.receipt, self.silent_since, self.failures, self.waiting = receipt, None, 0, None
        self.answered_here = True

    def dialing(self):
        self.answered_here = False

    def lost(self, exc):
        """The socket ended, or the dial reached no owner: re-dial after the returned pause, or give up."""
        if not self.answered_here:
            self.failures += 1
        self._silent(exc)
        attempt = f'; {self.failures} of {REDIAL_FAILURES} re-dials failed' if self.failures else ''
        print(f'Kanban owner lost after admission ({_cause(exc)}{attempt}); re-dialing to resume admission '
              f'{self.admission_id}, never resubmitting it', file=sys.stderr, flush=True)
        if self.failures >= REDIAL_FAILURES:
            self._give_up(exc, f'{self.failures} re-dials in a row reached no owner')
        return RECEIPT_POLL_S if self.answered_here else REDIAL_PAUSE_S * 2 ** (self.failures - 1)

    def unavailable(self, exc):
        """The owner says it is draining or starting (a restart, not a dead host): wait within budget."""
        self.failures = 0
        self._silent(exc)
        if exc.state != self.waiting:
            self.waiting = exc.state
            print(f'Kanban owner {exc.state} after admission; waiting up to {self.budget:.0f}s for it to answer '
                  f'admission {self.admission_id}', file=sys.stderr, flush=True)
        return REDIAL_PAUSE_S

    def _silent(self, exc):
        now = _monotonic()
        if self.silent_since is None:
            self.silent_since = now
        elif now - self.silent_since >= self.budget:
            self._give_up(exc, f'no receipt answer for {now - self.silent_since:.0f}s')

    def _give_up(self, exc, why):
        from hermes_cli.gateway_client import GatewayClientError
        raise GatewayClientError(
            f'Kanban owner lost after admission: {why} (last: {_cause(exc)}). The turn outcome is unknown '
            f'and the claim was not resubmitted; resume session {self.session_id} to check it') from exc


async def run(params, board_db):
    from websockets.exceptions import InvalidHandshake
    from hermes_cli.gateway_client import (
        connect_gateway, GatewayClientError, GatewayDisconnectedError, GatewayRPCError, GatewayUnavailableError,
        RUNNING_OWNER,
    )
    if os.environ.get('HERMES_TUI_GATEWAY_URL'):
        raise GatewayClientError('Kanban requires the assigned local profile owner')
    # Loaded now, not when the card settles: this process may outlive an update that rewrote the tree.
    import gateway.session_kanban
    deadline = _monotonic() + OWNER_WAIT_S
    claim = _Claim()
    while True:
        claim.dialing()
        receipt = lost = waiting = None
        try:
            # Before admission the first dial may start the owner, as it always has; after it, only
            # an owner that is up is dialed.
            dial = connect_gateway(endpoint=RUNNING_OWNER) if claim.admitted else connect_gateway()
            async with dial as client:
                # kanban.run subscribes this socket to the card's session, but only receipts matter
                # here: discard the turn's events, or they fill the client's bounded queue and its
                # reader stops, with the next receipt behind it.
                drain = asyncio.create_task(_discard(client.events))
                try:
                    receipt = await _settle(client, params, board_db, claim)
                except (GatewayDisconnectedError, TimeoutError) as exc:
                    if not claim.admitted:  # kanban.run's own answer is unknown: never resend it
                        raise
                    lost = exc
                finally:
                    drain.cancel()
        except GatewayUnavailableError as exc:
            if not claim.admitted:
                if exc.state not in _TRANSIENT_OWNER_STATES or _monotonic() >= deadline:
                    raise
                waiting = exc
            elif exc.state in _TRANSIENT_OWNER_STATES:
                waiting = exc
            else:
                lost = exc
        except GatewayRPCError:
            raise  # the owner's answer, never a transport loss
        except (GatewayClientError, OSError, InvalidHandshake) as exc:
            if not claim.admitted:
                raise
            lost = exc
        if receipt is not None:
            return _finish(receipt, board_db, params)
        if lost is not None:
            pause = claim.lost(lost)
        elif claim.admitted:
            pause = claim.unavailable(waiting)
        else:
            pause = REDIAL_PAUSE_S
        await _pause(pause)


async def _discard(events):
    while True:
        await events.get()


async def _settle(client, params, board_db, claim):
    """Submit the claim unless *claim* already holds its admission (it is never sent twice), then poll
    its receipt on *client* until it settles. A hang-up raises (``GatewayDisconnectedError``)."""
    if not claim.admitted:
        # Before kanban.run: once it answers, an exit here would amputate the turn it admitted.
        budget = resume_wait_budget()
        db = str(await asyncio.to_thread(Path(board_db).resolve))
        # The owner answers only after it checks the claim and writes its ``owner_admitted`` marker
        # in one board write transaction. The board (the cross-profile dispatch bus) can make that
        # wait out its busy timeout, so the 30 s default left the owner admitting an attempt whose
        # submitter had exited (the later-bound worker's submitter-liveness watcher ended it, and the
        # admission was left ``unknown``). The wait is still finite: once the marker commits,
        # ``enforce_max_runtime`` and TTL reclaim skip the run (``owner_reclaim_paused``), so an owner
        # that stalls after admitting is bounded only here. A hang-up fails the wait sooner.
        accepted = await client.rpc('kanban.run', _timeout=admission_answer_budget(), **dict(params, db=db))
        claim.admit(accepted['session_id'], accepted['receipt'], budget)
        print(f'Session: {claim.session_id}', file=sys.stderr)
    receipt = claim.receipt
    while receipt['status'] not in _SETTLED:
        await _pause(RECEIPT_POLL_S)
        receipt = await client.rpc('prompt.receipt', session_id=claim.session_id,
                                   admission_id=receipt['admission_id'])
        claim.answered(receipt)
    return receipt


async def _submit_claim(client, params, board_db):
    """Submit the claim on *client* and wait for its receipt to settle: the card's exit code."""
    return _finish(await _settle(client, params, board_db, _Claim()), board_db, params)


def _finish(receipt, board_db, params):
    from hermes_cli.gateway_client import GatewayClientError
    if receipt['status'] == 'unknown':
        raise GatewayClientError('Kanban execution unknown; retained by owner')
    from gateway.session_kanban import worker_result
    result = worker_result(board_db, params)
    # The attempt's reply ends the log, as a one-shot worker's did: crash diagnostics quote it.
    if result.get('last_output'):
        print(result['last_output'])
    return result.get('exit_code', 1)


def main():
    # The dead-worker sweep books this process's exit from the trailer in its log (it never
    # reaps us), so every exit path writes it — the same contract as a quiet one-shot worker.
    from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER
    rc = 1
    try:
        rc = _submit()
    finally:
        print(f'\n{KANBAN_WORKER_EXIT_TRAILER}{int(rc)}', file=sys.stderr, flush=True)
    return rc


def _submit():
    try:
        params = {'board': os.environ['HERMES_KANBAN_BOARD'], 'task_id': os.environ['HERMES_KANBAN_TASK'],
            'run_id': int(os.environ['HERMES_KANBAN_RUN_ID']), 'claim_lock': os.environ['HERMES_KANBAN_CLAIM_LOCK']}
        board_db = os.environ['HERMES_KANBAN_DB']
        # Owner startup must never inherit this worker's task/tool/goal identity.
        for key in list(os.environ):
            # The busy timeout is a host tuning knob, not task identity: the answer budget reads it.
            if key.startswith('HERMES_KANBAN_') and key not in ('HERMES_KANBAN_HOME', 'HERMES_KANBAN_BUSY_TIMEOUT_MS'):
                os.environ.pop(key)
        for key in ('HERMES_SESSION_SOURCE', 'HERMES_TENANT', 'TERMINAL_CWD'):
            os.environ.pop(key, None)
        return asyncio.run(run(params, board_db))
    except (KeyError, ValueError, OSError) as exc:
        print(f'Kanban owner refused: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

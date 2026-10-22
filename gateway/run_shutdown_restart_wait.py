"""The restart after-turn wait: defer ``stop()`` until in-flight work finishes or its cap elapses.

Split out of ``gateway/run_shutdown.py``; bound onto ``GatewayRunner`` via ``GatewayShutdownMixin``.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("gateway.run")


def _warn_lost_work(awaited: list, active: int, *, complete: bool = False) -> int:
    """Report the managed turns a restart's after-turn wait held for that ended LOST: the worker was
    gone before its turn finished (``ManagedWorker.lost``), so the outcome is unknown. A lost turn
    no longer counts as active work, and must never read as drained. Returns how many were lost."""
    lost = sorted(reason for reason in (getattr(worker, "lost", None) for worker in awaited) if reason)
    if lost and complete:
        logger.warning(
            "Restart deferred wait complete — %d of %d awaited work unit(s) LOST, not drained "
            "(managed worker lost before its turn finished: %s; turn outcome unknown); %d drained; "
            "proceeding to stop()", len(lost), active, ", ".join(lost), max(0, active - len(lost)),
        )
    elif lost:
        logger.warning(
            "Restart deferred wait: %d awaited work unit(s) LOST, not drained (managed worker lost "
            "before its turn finished: %s; turn outcome unknown)", len(lost), ", ".join(lost),
        )
    return len(lost)


class GatewayRestartWaitMixin:
    """``_await_active_work_before_restart``: hold a graceful restart until its in-flight turns end."""

    async def _await_active_work_before_restart(self) -> bool:
        """Wait for in-flight work before ``stop()`` so the requesting turn isn't force-interrupted.

        Wedged turns are excluded (restart is their remedy). True when every awaited unit finished,
        False when the cap elapsed, only wedged work remains, or an awaited managed turn was lost
        (caller proceeds to ``stop()``).
        """
        active = self._active_work_count()
        if active <= 0:
            return True
        if self._awaitable_work_count() <= 0:
            logger.warning(
                "Restart requested with %d active work unit(s), none awaitable "
                "(%d wedged past the inactivity timeout, %d in restart-safe external cron "
                "workers that outlive this process); skipping the after-turn wait and "
                "proceeding to stop()/drain", active,
                self._wedged_agent_count(), self._restart_safe_cron_count(),
            )
            return False
        timeout = float(getattr(self, "_restart_after_turn_timeout", 0.0) or 0.0)
        if timeout <= 0:
            logger.info(
                "Restart requested with %d active work unit(s); "
                "restart_after_turn_timeout=0 — entering stop()/drain immediately", active,
            )
            return False
        logger.info(
            "Restart requested with %d active work unit(s); "
            "deferring stop() until they finish (cap=%.0fs) so in-flight "
            "turns are not amputated (#77184)", active, timeout,
        )
        from gateway.session_runtime_workers import open_managed_workers
        # The managed turns this wait holds for. One whose worker is lost before its turn finishes (its
        # submitter gone, the worker killed; the turn's outcome left unknown) stops counting as active
        # work too, so it is reported as lost, never as drained.
        awaited = open_managed_workers(self)
        self._scale_to_zero_status("draining", "restart wait: status mark failed")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_status_at = 0.0
        while self._awaitable_work_count() > 0:
            now = loop.time()
            if now >= deadline:
                logger.warning(
                    "Restart after-turn wait timed out after %.0fs with %d "
                    "still active; proceeding to stop()/drain which may "
                    "interrupt remaining work (#77184)", timeout, self._active_work_count(),
                )
                _warn_lost_work(awaited, active)
                return False
            if (now - last_status_at) >= 30.0:
                logger.info(
                    "Restart deferred: waiting on %d active work unit(s) "
                    "(%d wedged and excluded, %d restart-safe and excluded; "
                    "%.0fs remaining before force drain): %s",
                    self._awaitable_work_count(), self._wedged_agent_count(),
                    self._restart_safe_cron_count(), deadline - now,
                    self._describe_active_work(),
                )
                self._scale_to_zero_status("draining", "restart wait: status mark failed")
                last_status_at = now
            await asyncio.sleep(0.1)
        if self._active_work_count() > 0:
            logger.warning(
                "Restart deferred wait: %d excluded work unit(s) remain "
                "(%d wedged, %d restart-safe external cron); proceeding to stop()/drain",
                self._active_work_count(), self._wedged_agent_count(),
                self._restart_safe_cron_count(),
            )
            _warn_lost_work(awaited, active)
            return False
        if _warn_lost_work(awaited, active, complete=True):
            return False
        logger.info("Restart deferred wait complete — active work drained; proceeding to stop()")
        return True

"""A tokenless ``auth: oauth`` MCP server in a process that cannot run a browser flow is attempted
once per process, reported by one WARNING with no traceback, and stays parked (tools unavailable)
until ``hermes mcp login <server>`` writes its tokens or a reconnect is requested.

Regression: after a gateway restart, errors.log gained ``MCP OAuth setup failed for '<server>'``
for each tokenless OAuth server every ``_PARKED_RETRY_INTERVAL`` (300 s). With no token file on disk the OAuth manager refuses
a non-interactive flow before any network I/O, so every timed self-probe of the auth-parked server
re-ran the same doomed setup and logged it again; the first discovery pass already reported the one
failure three times (setup warning, park warning, connect-failure warning).

stdlib ``unittest``; pytest collects it. Run from the checkout root so ``tools`` resolves to THIS
checkout: ``python -m unittest tests.tools.test_mcp_oauth_needs_login_park -v``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
NAME = "oauth-server"
CFG = {"url": "https://mcp.example.test/mcp", "auth": "oauth", "enabled": True}
WAKES = 4  # timed self-probe wakes observed before judging


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


class TokenlessOAuthServerPark(unittest.TestCase):
    def setUp(self):
        import tools

        if REPO not in Path(tools.__file__).resolve().parents:
            self.fail(f"tools imported from {tools.__file__}, not this checkout {REPO}; run from {REPO}")
        from tools import mcp_tool
        from tools import mcp_tool_config as _config
        from tools.mcp_oauth_manager import MCPOAuthManager
        from tools.mcp_tool import MCPServerTask

        self.home = Path(tempfile.mkdtemp(prefix="mcp-oauth-home-"))
        self.addCleanup(shutil.rmtree, self.home, True)
        old_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.home)
        self.addCleanup(lambda: os.environ.__setitem__("HERMES_HOME", old_home) if old_home is not None
                        else os.environ.pop("HERMES_HOME", None))

        self.configured = {NAME: dict(CFG)}
        for target, value in ((mcp_tool, {"_PARKED_RETRY_INTERVAL": 0.01, "_MAX_INITIAL_CONNECT_RETRIES": 0}),):
            for attr, new in value.items():
                patcher = mock.patch.object(target, attr, new)
                patcher.start()
                self.addCleanup(patcher.stop)
        for attr in ("_servers", "_server_scope_keys", "_server_connect_errors",
                     "_server_connect_retry_after", "_server_connect_failures"):
            patcher = mock.patch.dict(getattr(mcp_tool, attr))
            patcher.start()
            self.addCleanup(patcher.stop)
        connecting = set(mcp_tool._server_connecting)
        self.addCleanup(lambda: (mcp_tool._server_connecting.clear(), mcp_tool._server_connecting.update(connecting)))
        load = mock.patch.object(_config, "_load_mcp_config",
                                 lambda: {k: dict(v) for k, v in self.configured.items()})
        load.start()
        self.addCleanup(load.stop)

        # One OAuth setup attempt == one get_or_build_provider call (the real manager runs).
        self.attempts: list[str] = []
        real_build = MCPOAuthManager.get_or_build_provider

        def _counting_build(mgr, server_name, server_url, oauth_config):
            self.attempts.append(server_name)
            return real_build(mgr, server_name, server_url, oauth_config)

        build = mock.patch.object(MCPOAuthManager, "get_or_build_provider", _counting_build)
        build.start()
        self.addCleanup(build.stop)

        # Parked-wait outcomes ("self-probe" = the timed wake).
        self.outcomes: list[str] = []
        real_wait = MCPServerTask._wait_for_reconnect_or_shutdown

        async def _recording_wait(task, timeout=None):
            outcome = await real_wait(task, timeout=timeout)
            if task.name == NAME:
                self.outcomes.append(outcome)
            return outcome

        wait = mock.patch.object(MCPServerTask, "_wait_for_reconnect_or_shutdown", _recording_wait)
        wait.start()
        self.addCleanup(wait.stop)

        # The network boundary: no test may open an outbound connection. A transport reached here
        # is refused like an unreachable host.
        self.transport_calls: list[str] = []

        def _refused(task, *a, **kw):
            self.transport_calls.append(task.name)
            raise ConnectionRefusedError("test transport: no network")

        for attr in ("_streamable_http_transport", "_sse_transport"):
            patcher = mock.patch.object(MCPServerTask, attr, _refused)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.capture = _Capture()
        logging.getLogger().addHandler(self.capture)
        self.addCleanup(logging.getLogger().removeHandler, self.capture)

    # -- helpers ---------------------------------------------------------------------------------
    def _warnings_for(self, name: str):
        return [r for r in self.capture.records
                if r.levelno >= logging.WARNING and name in r.getMessage()]

    def _write_tokens(self):
        from tools.mcp_oauth import HermesTokenStorage

        path = HermesTokenStorage(NAME)._tokens_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"access_token": "a", "token_type": "Bearer", "refresh_token": "r"}))

    async def _discover_and_park(self):
        """The real discovery pass every process runs at session start, non-interactive."""
        from tests.waiting import async_wait_until
        from tools import mcp_tool
        from tools.mcp_tool_discovery import _discover_all

        await _discover_all({NAME: dict(CFG)})
        server = mcp_tool._servers.get(NAME)
        if server is None:
            self.fail("the auth-failed server must be adopted as a recoverable park")
        await async_wait_until(lambda: server._was_parked, timeout=10, what="server parked")
        self.assertFalse(server._task.done(), "a parked server's run task must stay alive (revivable)")
        self.assertEqual(server.session, None)
        self.assertEqual(list(getattr(server, "_registered_tool_names", []) or []), [],
                         "a parked server's tools stay unavailable")
        return server

    async def _wait_wakes(self, count: int, *, since: int = 0):
        from tests.waiting import async_wait_until

        await async_wait_until(lambda: self.outcomes[since:].count("self-probe") >= count,
                               timeout=10, what=f"{count} timed self-probe wake(s)")

    def _run(self, scenario):
        from tools.mcp_oauth import suppress_interactive_oauth

        with suppress_interactive_oauth():  # gateway / managed worker: no browser, no TTY
            asyncio.run(scenario())

    # -- the defect --------------------------------------------------------------------------------
    def test_tokenless_server_is_attempted_once_and_reported_once_per_process(self):
        async def scenario():
            from tools.mcp_tool_discovery import _select_new_servers

            server = await self._discover_and_park()
            for _ in range(3):  # three later session starts in the same process
                _select_new_servers({NAME: dict(CFG)})
            await self._wait_wakes(WAKES)
            attempts = len(self.attempts)
            await server.shutdown()
            return attempts

        holder = {}

        async def wrapped():
            holder["attempts"] = await scenario()

        self._run(wrapped)
        with self.subTest("never reaches the network"):
            self.assertEqual(self.transport_calls, [], "a tokenless server must never reach the network")
        with self.subTest("attempted once per process"):
            self.assertEqual(
                holder["attempts"], 1,
                f"OAuth setup for a tokenless server ran {holder['attempts']}x across {WAKES} timed "
                "self-probes; nothing can change until `hermes mcp login` writes its tokens")
        lines = self._warnings_for(NAME)
        with self.subTest("reported by one WARNING line"):
            self.assertEqual(
                len(lines), 1,
                f"expected exactly one WARNING for the parked server, got {len(lines)}:\n"
                + "\n".join(f"  {r.name}: {r.getMessage()[:150]}" for r in lines))
            self.assertIn(f"hermes mcp login {NAME}", lines[0].getMessage())
        with self.subTest("no traceback"):
            self.assertFalse([r for r in self.capture.records if NAME in r.getMessage() and r.exc_info],
                             "no traceback may be logged for a missing login")

    # -- pins: what must keep working ------------------------------------------------------------------
    def test_login_landing_revives_on_the_next_timed_self_probe(self):
        """`hermes mcp login` in another process writes the token file; no restart, no reconnect."""

        async def scenario():
            server = await self._discover_and_park()
            await self._wait_wakes(2)
            before, mark = len(self.attempts), len(self.outcomes)
            self._write_tokens()
            from tests.waiting import async_wait_until

            await async_wait_until(lambda: len(self.attempts) > before, timeout=10,
                                   what="a re-attempt after the token file appeared")
            self.assertNotIn("reconnect", self.outcomes[mark:], "revival must come from the timed probe")
            await server.shutdown()

        self._run(scenario)
        self.assertGreaterEqual(len(self.transport_calls), 1,
                                "with tokens on disk the revival proceeds to the transport")

    def test_explicit_reconnect_still_retries_a_tokenless_server(self):
        """/reload-mcp or a dashboard re-auth sets ``_reconnect_event``: always honoured."""

        async def scenario():
            server = await self._discover_and_park()
            before = len(self.attempts)
            server._reconnect_event.set()
            from tests.waiting import async_wait_until

            await async_wait_until(lambda: len(self.attempts) > before, timeout=10,
                                   what="a re-attempt after an explicit reconnect")
            await server.shutdown()

        self._run(scenario)

    def test_auth_park_with_cached_tokens_keeps_its_timed_self_probe(self):
        """Upstream intent (f7b26efc2f): a server WITH tokens that failed auth (throttled token
        endpoint, expired refresh) is still retried every interval."""
        from tools.mcp_oauth import OAuthNonInteractiveError
        from tools.mcp_tool import MCPServerTask

        self._write_tokens()
        calls = []

        class _Task(MCPServerTask):
            async def _run_http(self, config):
                calls.append(config["url"])
                raise OAuthNonInteractiveError("refresh token rejected; browser authorization required")

        async def scenario():
            task = _Task(NAME)
            run = asyncio.ensure_future(task.run(dict(CFG)))
            await self._wait_wakes(3)
            task._shutdown_event.set()
            task._reconnect_event.set()
            await asyncio.wait_for(run, 10)

        self._run(scenario)
        self.assertGreaterEqual(len(calls), 3, f"timed self-probe stopped retrying a token-holding server: {calls}")

    def test_other_connect_failures_still_warn_from_discovery(self):
        """The one-line rule is for a missing login only: an unreachable server keeps its WARNING."""
        from tools.mcp_tool_discovery import _discover_all

        self.configured = {"unreachable": {"url": "https://down.example.test/mcp", "skip_preflight": True}}

        async def scenario():
            from tools import mcp_tool

            await _discover_all({"unreachable": dict(self.configured["unreachable"])})
            server = mcp_tool._servers.get("unreachable")
            if server is not None:
                await server.shutdown()

        self._run(scenario)
        self.assertTrue(
            [r for r in self._warnings_for("unreachable") if "Failed to connect to MCP server" in r.getMessage()],
            "a non-auth connect failure must still be reported at WARNING by discovery")


if __name__ == "__main__":
    unittest.main()

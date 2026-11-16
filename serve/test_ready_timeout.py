"""#1527: the READY-read timeout (#1317) ends the engine, and the request that was loading it must end too -
a 503 with the reason, not a connection that never answers (or a misleading 400)."""
import os
import contextlib
import subprocess
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import json
import urllib.request
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server                                            # noqa: E402
from serve.server import StrataEngine, Service, ChatTemplate, ByteTokenizer, serve  # noqa: E402

# never prints READY: the READY read times out and the process is killed
SILENT = r'''
import time
time.sleep(3600)
'''

# first run answers READY + one request; after --hang is touched a restart stays silent forever
HANG_ON_RESTART = r'''
import sys, time
from pathlib import Path
hang = Path(sys.argv[sys.argv.index('--hang') + 1])
if hang.exists():
    time.sleep(3600)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith(('GEN', 'BGEN')):
        print('T 90\nDONE 1 1 0 1 stop', flush=True)
'''


@unittest.skipIf(os.name == "nt", "the .cmd wrapper keeps the pipe open after the kill (only the wrapper dies), so the "
                                  "close of its stdout waits for the sleeping child; a real engine is one process")
class ReadyTimeout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._patchers = [mock.patch.object(server, 'ENGINE_READY_S', 0.5),
                          mock.patch.object(StrataEngine, 'RESTART_RETRY_S', 0.0)]
        for p in self._patchers:
            p.start(); self.addCleanup(p.stop)

    def make_script(self, name, code):
        py = self.root / (name + '.py')
        if os.name == 'nt':
            py.write_text(code); script = self.root / (name + '.cmd')
            script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            script = py; py.write_text('#!' + sys.executable + '\n' + code); py.chmod(0o700)
        return str(script)

    def service(self, script, extra_args=()):
        eng = StrataEngine(script, list(extra_args), lazy=True)
        self.addCleanup(eng.close)
        return Service(eng, ByteTokenizer(), ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))

    def post(self, base):
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8}
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()

    def assert503(self, base):
        t0 = time.monotonic()
        try:
            self.post(base)
            self.fail('a request on an engine that never reports READY must not succeed')
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 503)
            body = e.read().decode()
            self.assertIn('did not report READY', body)
            self.assertIn('next request restarts it', body)
        self.assertLess(time.monotonic() - t0, 60)

    @contextlib.contextmanager
    def captured_start(self, script, configure=None, before_spawn=None, log=None):
        """Real child/READY threads, with finite fault injection and owned cleanup."""
        eng = StrataEngine(script, [], log=log, lazy=True)
        children, readers, extra_fds = [], [], []
        prevented_starts = []
        real_popen, real_thread = server.popen, threading.Thread
        def spawn(*args, **kwargs):
            if before_spawn is not None:
                before_spawn(eng, children, readers)
            # Fault cases must not leak a successor even on the old implementation.
            if configure is not None and children:
                raise AssertionError("second spawn before captured retirement")
            if configure == "blocked_reader":
                read_fd, write_fd = os.pipe()
                extra_fds.append(write_fd)
                kwargs["stdout"] = write_fd
            proc = real_popen(*args, **kwargs)
            if configure == "blocked_reader":
                proc.stdout = os.fdopen(read_fd, "r", encoding="utf-8")
            saved = (proc, proc.wait, proc.kill, proc.stdin, proc.stdout, eng.log)
            children.append(saved)
            if configure == "wait5":
                proc.kill = mock.Mock()  # Genuine child remains alive.
                proc.wait = mock.Mock(side_effect=subprocess.TimeoutExpired("captured", 5))
            elif configure == "wait60":
                original_wait = proc.wait
                def wait(timeout=None):
                    if timeout == 60:
                        raise subprocess.TimeoutExpired("captured", timeout)
                    return original_wait(timeout=timeout)
                proc.wait = wait
            return proc
        def thread(*args, **kwargs):
            result = real_thread(*args, **kwargs)
            if kwargs.get("target") == StrataEngine._ready_pump:
                readers.append(result)
                if configure == "wait60":
                    prevented = mock.Mock(side_effect=RuntimeError("injected READY reader start failure"))
                    prevented_starts.append((result, result.start, prevented))
                    result.start = prevented
            return result
        try:
            with mock.patch.object(server, "popen", side_effect=spawn), \
                 mock.patch.object(server.threading, "Thread", side_effect=thread):
                yield eng, children, readers, extra_fds
        finally:
            for fd in extra_fds:
                os.close(fd)
            for proc, wait, kill, stdin, stdout, log_handle in children:
                proc.wait, proc.kill = wait, kill
                if proc.poll() is None:
                    kill()
                wait(timeout=5)
            for reader, original_start, prevented in prevented_starts:
                was_prevented = reader.start is prevented and prevented.called
                reader.start = original_start
                if was_prevented and reader.ident is None:
                    reader.start()  # Only our called raising mock prevented this start.
            for reader in readers:
                if reader.ident is not None:
                    reader.join(5)
                    self.assertFalse(reader.is_alive(), "captured READY reader leaked")
            eng.close()
            for proc, wait, kill, stdin, stdout, log_handle in children:
                stdin.close()
                stdout.close()
                if log_handle not in (None, subprocess.DEVNULL):
                    log_handle.close()

    def test_failed_ready_wait5_retains_child_without_second_spawn(self):
        with self.captured_start(self.make_script("wait5", SILENT), "wait5") as (eng, children, readers, _):
            with self.assertRaises(server.EngineStuck):
                eng.restart(tries=2)
            self.assertEqual(len(children), 1)
            self.assertIs(eng.proc, children[0][0])
            self.assertIsNone(eng.proc.poll())
            self.assertIs(eng.pump, readers[0])
            self.assertTrue(eng.pump.is_alive())
            self.assertFalse(eng.proc.stdin.closed)
            self.assertFalse(eng.proc.stdout.closed)
            self.assertEqual(eng.__dict__.get("gen", 0), 0)

    def test_failed_ready_start_wait60_retains_child_without_second_spawn(self):
        with self.captured_start(self.make_script("wait60", SILENT), "wait60") as (eng, children, readers, _):
            with self.assertRaises(server.EngineStuck):
                eng.restart(tries=2)
            self.assertEqual(len(children), 1)
            self.assertIs(eng.proc, children[0][0])
            self.assertIsNone(eng.proc.poll())
            self.assertIs(eng.pump, readers[0])
            self.assertIsNone(eng.pump.ident)
            self.assertFalse(eng.proc.stdin.closed)
            self.assertFalse(eng.proc.stdout.closed)
            self.assertEqual(eng.__dict__.get("gen", 0), 0)

    def test_exited_child_blocked_ready_reader_fences_retry(self):
        with self.captured_start(self.make_script("exited", "raise SystemExit(1)\n"), "blocked_reader") as state:
            eng, children, readers, fds = state
            result, finished = [], threading.Event()
            def restart():
                try:
                    eng.restart(tries=2)
                except BaseException as error:
                    result.append(error)
                finally:
                    finished.set()
            caller = threading.Thread(target=restart, daemon=True)
            with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .05), \
                 mock.patch.object(server, "ENGINE_TERM_WAIT_S", .1), \
                 mock.patch.object(server, "ENGINE_KILL_WAIT_S", .1):
                caller.start()
                try:
                    self.assertTrue(finished.wait(3), "caller blocked on captured stdout.close")
                    self.assertEqual(len(result), 1)
                    self.assertIsInstance(result[0], server.EngineStuck)
                    self.assertEqual(len(children), 1)
                    self.assertIs(eng.proc, children[0][0])
                    self.assertIsNotNone(eng.proc.poll())
                    self.assertIs(eng.pump, readers[0])
                    self.assertTrue(eng.pump.is_alive())
                    self.assertFalse(eng.proc.stdout.closed)
                    os.fstat(eng.proc.stdout.fileno())
                finally:
                    for fd in fds:
                        os.close(fd)
                    fds.clear()
                    caller.join(5)
                    self.assertFalse(caller.is_alive(), "restart caller leaked")

    def test_fully_retired_start_allows_outer_and_internal_retry(self):
        for internal in (False, True):
            with self.subTest(internal=internal):
                marker = self.root / ("internal" if internal else "outer")
                script = self.make_script(marker.name, "import sys\nfrom pathlib import Path\n"
                    + "p = Path(" + repr(str(marker)) + ")\n"
                    + "if not p.exists():\n    p.touch()\n    raise SystemExit(" + ("0" if internal else "1") + ")\n"
                    + "print('READY 4096 stop', flush=True)\n"
                    + "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
                def before_spawn(eng, children, readers):
                    if children:
                        self.assertEqual(len(children), 1)
                        proc, _, _, stdin, stdout, log_handle = children[0]
                        self.assertIsNotNone(proc.poll())
                        self.assertTrue(stdin.closed)
                        self.assertTrue(stdout.closed)
                        self.assertTrue(log_handle.closed)
                        self.assertFalse(readers[0].is_alive())
                        self.assertIsNone(eng.__dict__.get("_close_owner"))
                with self.captured_start(script, before_spawn=before_spawn,
                                         log=str(self.root / (marker.name + ".log"))) as (eng, children, readers, _):
                    eng.restart(tries=2)
                    self.assertEqual(len(children), 2)
                    self.assertEqual(eng.max_context, 4096)
                    self.assertIs(eng.proc, children[1][0])
                    self.assertFalse(readers[0].is_alive())

    def test_direct_constructor_fault_never_selects_split_fallback(self):
        for fault in ("wait5", "wait60"):
            with self.subTest(fault=fault):
                failed = None
                try:
                    with self.captured_start(self.make_script("direct_" + fault, SILENT), fault) as state:
                        _, children, readers, _ = state
                        cfg = {"args": [], "split_fallback": {"exe": "/must-not-start"}}
                        with mock.patch.object(server, "engine_args", lambda cfg, *a, **k: cfg.get("args", [])), \
                             mock.patch.object(server, "mtp_retry_args", lambda *a: None), \
                             mock.patch.object(server, "nodraft_retry_args", lambda *a: None):
                            with self.assertRaises(server.EngineStuck) as caught:
                                server.start_engine_or_fallback(cfg, self.make_script("direct_" + fault, SILENT),
                                                                [], {}, False, None)
                        failed = caught.exception.startup_engine
                        self.assertEqual(len(children), 1)
                        self.assertIs(failed.proc, children[0][0])
                        self.assertFalse(failed._startup_attempt.retired)
                        self.assertIsNone(failed.proc.poll())
                        self.assertIs(failed.pump, readers[0])
                finally:
                    if failed is not None:
                        failed.close()

    def test_competing_restart_is_refused_before_popen(self):
        script = self.make_script("concurrent", "import sys\nprint('READY 4096 stop', flush=True)\n"
                                  "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
        with self.captured_start(script) as (eng, children, _, _):
            entered, release = threading.Event(), threading.Event()
            popen, errors = server.popen, []
            def blocked_spawn(*args, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise AssertionError("startup barrier not released")
                return popen(*args, **kwargs)
            def start():
                try:
                    eng.restart(tries=1)
                except BaseException as error:
                    errors.append(error)
            caller = threading.Thread(target=start, daemon=True)
            with mock.patch.object(server, "popen", side_effect=blocked_spawn):
                caller.start()
                try:
                    self.assertTrue(entered.wait(5))
                    with self.assertRaises(server.EngineStuck):
                        eng.restart(tries=1)
                    self.assertEqual(len(children), 0)
                finally:
                    release.set()
                    caller.join(5)
            self.assertFalse(caller.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(children), 1)
            self.assertIs(eng._epoch.proc, children[0][0])

    def test_cancelled_prepared_ready_cannot_publish_successor_epoch(self):
        script = self.make_script("stale", "import sys\nprint('READY 4096 stop', flush=True)\n"
                                  "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
        with self.captured_start(script) as (eng, children, _, _):
            entered, release = threading.Event(), threading.Event()
            errors = []
            original_start = type(threading.Thread(target=lambda: None)).start
            def start_thread(thread):
                if (getattr(getattr(thread, "_target", None), "__name__", "") == "_startup_pump"
                        and not entered.is_set()):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("READY publication barrier not released")
                return original_start(thread)
            def start():
                try:
                    eng.restart(tries=1)
                except BaseException as error:
                    errors.append(error)
            # captured_start's Thread factory returns actual Thread objects.
            # Use the class of an actual unstarted Thread rather than the patched factory.
            caller = threading.Thread(target=start, daemon=True)
            with mock.patch.object(type(caller), "start", start_thread):
                caller.start()
                try:
                    self.assertTrue(entered.wait(5))
                    with self.assertRaises(server.EngineStuck):
                        eng.close()
                    with self.assertRaises(server.EngineStuck):
                        eng.restart(tries=1)
                    self.assertEqual(len(children), 1)
                    self.assertEqual(eng.__dict__.get("gen", 0), 0)
                finally:
                    release.set()
                    caller.join(5)
            self.assertFalse(caller.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], server.EngineDied)
            self.assertIsNone(eng.proc)
            self.assertIsNone(eng.__dict__.get("_startup_attempt"))
            self.assertEqual(eng.__dict__.get("gen", 0), 0)
            eng.restart(tries=1)
            self.assertEqual(len(children), 2)
            self.assertIs(eng._epoch.proc, children[1][0])
            self.assertIsNot(eng._epoch.proc, children[0][0])
            self.assertTrue(eng.pump.is_alive())

    def test_alternate_and_vram_recursion_wait_for_actual_prior_retirement(self):
        for vram in (False, True):
            with self.subTest(vram=vram):
                script = self.make_script("recursive_" + str(vram),
                    "import sys\nif '--first' in sys.argv: raise SystemExit(1)\n"
                    "print('READY 4096 stop', flush=True)\n"
                    "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
                def before_spawn(eng, children, readers):
                    if children:
                        self.assertEqual(len(children), 1)
                        old, _, _, stdin, stdout, _ = children[0]
                        self.assertIsNotNone(old.poll())
                        self.assertTrue(stdin.closed)
                        self.assertTrue(stdout.closed)
                        self.assertFalse(readers[0].is_alive())
                        self.assertIsNone(eng.__dict__.get("_close_owner"))
                with self.captured_start(script, before_spawn=before_spawn) as (eng, children, _, _):
                    # Only the native reason classifier boundary is controlled;
                    # all startup/retirement/recursive constructor methods are real.
                    with mock.patch.object(server.vram_fit, "engine_stop_reason",
                                           return_value="fixture native VRAM reason" if vram else None):
                        eng.__init__(script, ["--first"], retry_args=None if vram else ["--fallback"],
                                     retry_vram_args=["--fallback"] if vram else None)
                    self.assertEqual(len(children), 2)
                    self.assertIn("--first", children[0][0].args)
                    self.assertIn("--fallback", children[1][0].args)
                    self.assertIs(eng._epoch.proc, children[1][0])
                    self.assertEqual(eng.gen, 1)

    def test_unresolved_child_fences_both_recursive_fallback_branches(self):
        for vram in (False, True):
            with self.subTest(vram=vram):
                with self.captured_start(self.make_script("blocked_recursion_" + str(vram), SILENT), "wait5") as state:
                    eng, children, _, _ = state
                    with mock.patch.object(server.vram_fit, "engine_stop_reason",
                                           side_effect=AssertionError("classified before retirement")):
                        with self.assertRaises(server.EngineStuck):
                            eng.__init__(eng.spawn[0], ["--first"], retry_args=None if vram else ["--fallback"],
                                         retry_vram_args=["--fallback"] if vram else None)
                    self.assertEqual(len(children), 1)
                    self.assertIs(eng.proc, children[0][0])
                    self.assertIsNone(eng.proc.poll())
                    self.assertFalse(eng._startup_attempt.retired)

    def test_actual_retired_split_child_allows_normal_fallback(self):
        first = self.make_script("split_refusal", "raise SystemExit(1)\n")
        normal = self.make_script("normal_fallback", "import sys\nprint('READY 4096 stop', flush=True)\n"
                                  "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
        def before_spawn(eng, children, readers):
            if children:
                self.assertEqual(len(children), 1)
                old, _, _, stdin, stdout, _ = children[0]
                self.assertIsNotNone(old.poll())
                self.assertTrue(stdin.closed)
                self.assertTrue(stdout.closed)
                self.assertFalse(readers[0].is_alive())
        with self.captured_start(first, before_spawn=before_spawn) as (_, children, _, _):
            cfg = {"args": [], "split_fallback": {"exe": normal, "args": []}}
            with mock.patch.object(server, "engine_args", lambda cfg, *a, **k: cfg.get("args", [])), \
                 mock.patch.object(server, "mtp_retry_args", lambda *a: None), \
                 mock.patch.object(server, "nodraft_retry_args", lambda *a: None), \
                 mock.patch.object(server, "with_component_engine", side_effect=lambda cfg: cfg), \
                 mock.patch.object(server, "child_env", return_value={}), \
                 mock.patch.object(server, "effort_end_args", return_value=None), \
                 mock.patch.object(server, "configure_batch_telemetry_env"):
                engine = server.start_engine_or_fallback(cfg, first, [], {}, False, None)
            try:
                self.assertEqual(len(children), 2)
                self.assertTrue(engine.info["split_fallback"])
                self.assertIs(engine.proc, children[1][0])
                self.assertIs(engine._epoch.proc, engine.proc)
            finally:
                engine.close()

    def test_completed_external_close_retires_later_reserved_restart(self):
        script = self.make_script("close_before_retire", "import sys\nprint('READY 4096 stop', flush=True)\n"
                                  "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
        old, close_errors, restart_errors = {}, [], []
        close_at_tail, release_close = threading.Event(), threading.Event()
        reserved, release_restart = threading.Event(), threading.Event()
        def before_spawn(eng, children, readers):
            if children:
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(old["proc"].poll())
                self.assertTrue(old["proc"].stdin.closed)
                self.assertTrue(old["proc"].stdout.closed)
                self.assertTrue(old["log"].closed)
                self.assertFalse(old["reader"].is_alive())
                self.assertFalse(readers[0].is_alive())
                self.assertFalse(old["close_owner"][4].is_alive())
                self.assertIsNone(eng.__dict__.get("_close_owner"))
        with mock.patch.dict(os.environ, {"STRATA_KEEP_LEFTOVERS": "1"}), \
             self.captured_start(script, before_spawn=before_spawn,
                                 log=str(self.root / "close_before_retire.log")) as (eng, children, readers, _):
            eng.restart(tries=1)
            old.update(proc=eng.proc, reader=eng.pump, log=eng.log, gen=eng.gen)
            self.assertIsNone(eng.__dict__.get("_startup_attempt"))
            actual_stdout_close, actual_retire = eng.proc.stdout.close, eng._retire_startup
            def close_stdout():
                actual_stdout_close()
                old["close_owner"] = eng._close_owner
                close_at_tail.set()
                if not release_close.wait(5):
                    raise AssertionError("external close tail barrier not released")
            def retire(attempt):
                if not reserved.is_set():
                    self.assertIs(attempt.proc, old["proc"])
                    self.assertIs(attempt.ready_reader, old["reader"])
                    reserved.set()
                    if not release_restart.wait(5):
                        raise AssertionError("reserved restart barrier not released")
                return actual_retire(attempt)
            def close():
                try:
                    eng.close()
                except BaseException as error:
                    close_errors.append(error)
            def restart():
                try:
                    eng.restart(tries=1)
                except BaseException as error:
                    restart_errors.append(error)
            closer = threading.Thread(target=close, daemon=True)
            starter = threading.Thread(target=restart, daemon=True)
            with mock.patch.object(old["proc"].stdout, "close", side_effect=close_stdout), \
                 mock.patch.object(eng, "_retire_startup", side_effect=retire):
                try:
                    closer.start()
                    self.assertTrue(close_at_tail.wait(5))
                    starter.start()
                    self.assertTrue(reserved.wait(5))
                    self.assertEqual(len(children), 1)
                    release_close.set()
                    closer.join(5)
                    self.assertFalse(closer.is_alive())
                    self.assertEqual(close_errors, [])
                    # Genuine physical observations precede allowing any new spawn.
                    before_spawn(eng, children, readers)
                    release_restart.set()
                    starter.join(5)
                    self.assertFalse(starter.is_alive())
                finally:
                    release_close.set()
                    release_restart.set()
                    for worker in (closer, starter):
                        if worker.ident is not None:
                            worker.join(5)
                            self.assertFalse(worker.is_alive())
            self.assertEqual(restart_errors, [])
            self.assertEqual(len(children), 2)
            self.assertIs(eng.proc, children[1][0])
            self.assertIs(eng._epoch.proc, children[1][0])
            self.assertEqual(eng.gen, old["gen"] + 1)
            self.assertEqual(eng._epoch.gen, eng.gen)
            self.assertIsNot(eng.pump, old["reader"])
            self.assertTrue(eng.pump.is_alive())
            self.assertFalse(eng.proc.stdin.closed)
            self.assertFalse(eng.proc.stdout.closed)
            self.assertIsNot(eng.log, old["log"])
            self.assertFalse(eng.log.closed)

    def test_retire_read_before_close_tail_rechecks_real_receipt_after_strict_guard(self):
        script = self.make_script("close_after_retire_read", "import sys\nprint('READY 4096 stop', flush=True)\n"
                                  "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
        old, close_errors, restart_errors, guard_errors = {}, [], [], []
        close_at_tail, release_close = threading.Event(), threading.Event()
        read_old, release_expected_close = threading.Event(), threading.Event()
        def before_spawn(eng, children, readers):
            if children:
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(old["proc"].poll())
                self.assertTrue(old["proc"].stdin.closed)
                self.assertTrue(old["proc"].stdout.closed)
                self.assertTrue(old["log"].closed)
                self.assertFalse(old["reader"].is_alive())
                self.assertFalse(readers[0].is_alive())
                self.assertFalse(old["close_owner"][4].is_alive())
                self.assertIsNone(eng.__dict__.get("_close_owner"))
        with mock.patch.dict(os.environ, {"STRATA_KEEP_LEFTOVERS": "1"}), \
             self.captured_start(script, before_spawn=before_spawn,
                                 log=str(self.root / "close_after_retire_read.log")) as (eng, children, readers, _):
            eng.restart(tries=1)
            old.update(proc=eng.proc, reader=eng.pump, log=eng.log, gen=eng.gen)
            self.assertIsNone(eng.__dict__.get("_startup_attempt"))
            actual_stdout_close, actual_close = eng.proc.stdout.close, eng.close
            def close_stdout():
                actual_stdout_close()
                old["close_owner"] = eng._close_owner
                close_at_tail.set()
                if not release_close.wait(5):
                    raise AssertionError("external close tail barrier not released")
            def captured_close(*, _expected_proc=None, _startup=None):
                intercepted = _expected_proc is old["proc"] and _startup is not None and not read_old.is_set()
                if intercepted:
                    self.assertFalse(_startup.retired)
                    self.assertIs(_startup.proc, old["proc"])
                    self.assertIs(_startup.ready_reader, old["reader"])
                    read_old.set()  # Actual _retire_startup already read false/oldproc.
                    if not release_expected_close.wait(5):
                        raise AssertionError("expected close barrier not released")
                try:
                    return actual_close(_expected_proc=_expected_proc, _startup=_startup)
                except server.EngineStuck as error:
                    if intercepted:
                        guard_errors.append(error)  # The STRONG guard still actually raises.
                    raise
            def close():
                try:
                    eng.close()
                except BaseException as error:
                    close_errors.append(error)
            def restart():
                try:
                    eng.restart(tries=1)
                except BaseException as error:
                    restart_errors.append(error)
            closer = threading.Thread(target=close, daemon=True)
            starter = threading.Thread(target=restart, daemon=True)
            with mock.patch.object(old["proc"].stdout, "close", side_effect=close_stdout), \
                 mock.patch.object(eng, "close", side_effect=captured_close):
                try:
                    closer.start()
                    self.assertTrue(close_at_tail.wait(5))
                    starter.start()
                    self.assertTrue(read_old.wait(5))
                    self.assertEqual(len(children), 1)
                    release_close.set()
                    closer.join(5)
                    self.assertFalse(closer.is_alive())
                    self.assertEqual(close_errors, [])
                    before_spawn(eng, children, readers)
                    release_expected_close.set()
                    starter.join(5)
                    self.assertFalse(starter.is_alive())
                finally:
                    release_close.set()
                    release_expected_close.set()
                    for worker in (closer, starter):
                        if worker.ident is not None:
                            worker.join(5)
                            self.assertFalse(worker.is_alive())
            self.assertEqual(len(guard_errors), 1)
            self.assertEqual(restart_errors, [])
            self.assertEqual(len(children), 2)
            self.assertIs(eng.proc, children[1][0])
            self.assertIs(eng._epoch.proc, children[1][0])
            self.assertEqual(eng.gen, old["gen"] + 1)
            self.assertEqual(eng._epoch.gen, eng.gen)
            self.assertIsNot(eng.pump, old["reader"])
            self.assertTrue(eng.pump.is_alive())
            self.assertFalse(eng.proc.stdin.closed)
            self.assertFalse(eng.proc.stdout.closed)
            self.assertIsNot(eng.log, old["log"])
            self.assertFalse(eng.log.closed)

    def test_ready_timeout_on_lazy_load_answers_the_request(self):
        # lazy start: restart()'s own bookkeeping must not crash on the missing slot_cv either (#1527)
        svc = self.service(self.make_script('silent', SILENT))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown); self.addCleanup(httpd.server_close)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        self.assert503(base)
        self.assertFalse(svc.engine.starting)

    def test_ready_timeout_on_restart_answers_the_request(self):
        # the reporter's case: the engine had answered once, then a restart hangs on the READY read
        hang = self.root / 'hang'
        svc = self.service(self.make_script('hang_on_restart', HANG_ON_RESTART), ('--hang', str(hang)))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown); self.addCleanup(httpd.server_close)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        status, _ = self.post(base)                              # first request: the engine works
        self.assertEqual(status, 200)
        svc.engine.proc.kill(); svc.engine.proc.wait(10)         # it dies between requests
        hang.touch()                                             # and its restart never reports READY
        self.assert503(base)
        self.assertFalse(svc.engine.starting)


class SpawnFailure(unittest.TestCase):
    """The engine program cannot be started at all (missing, not executable, or blocked by Windows - #735): the
    request that was loading it gets a 503 with the reason, not a dropped connection."""
    BODIES = {'/v1/chat/completions': {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8},
              '/v1/messages': {'model': 'strata', 'max_tokens': 8, 'messages': [{'role': 'user', 'content': 'hi'}]},
              '/v1/responses': {'input': 'hi', 'max_output_tokens': 8}}

    def test_missing_engine_answers_503_and_leaves_nothing_running(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        eng = StrataEngine(str(root / 'no-such-engine'), [], log=str(root / 'engine.log'), lazy=True)
        svc = Service(eng, ByteTokenizer(), ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown); self.addCleanup(httpd.server_close)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        before = set(threading.enumerate())
        cases = list(self.BODIES.items()) + [('/v1/chat/completions',
                                              {**self.BODIES['/v1/chat/completions'], 'stream': True})]
        for path, body in cases:
            with self.subTest(path=path, stream=body.get('stream', False)):
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(req, timeout=30)
                self.assertEqual(cm.exception.code, 503)
                self.assertIn('the engine could not be started', cm.exception.read().decode())
                self.assertTrue(eng.log.closed)                 # our handle on the engine log is given back
        # the start narrator ("still starting ... please wait") ends with the failed start
        deadline = time.monotonic() + 5
        while (narrators := [t for t in set(threading.enumerate()) - before if '(narrate_start)' in t.name]) \
                and time.monotonic() < deadline:
            time.sleep(0.1)
        self.assertEqual(narrators, [])


if __name__ == '__main__':
    unittest.main()

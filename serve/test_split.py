"""Split serving (serve/split.py): the config, the decision, the hand-over with two fake engines, and every fallback.
The engine-side proof (the continuation after a hand-over equals a single engine's) is tests/split_serve_exact.py."""
import os
import contextlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import tempfile
import threading
import unittest

from serve import split as sp
from serve.server import EngineDied, SessionRefused
from serve.split import SplitConfigError, SplitEngine, check_cfg, common_prefix, role_cfg, split_spec


class Config(unittest.TestCase):
    def test_off_without_the_key(self):
        self.assertIsNone(split_spec({}))
        self.assertIsNone(split_spec({"split_serve": False}))
        self.assertIsNone(split_spec({"split_serve": None}))

    def test_cards_and_defaults(self):
        s = split_spec({"split_serve": {"prefill_gpu": 0, "decode_gpu": "1"}})
        self.assertEqual((s["prefill_gpu"], s["decode_gpu"]), (0, 1))
        self.assertEqual(s["min_prompt_tokens"], sp.MIN_TOKENS)
        self.assertIs(s["shared_arena"], True)
        self.assertEqual((s["prefill_args"], s["decode_args"]), ([], []))

    def test_mistakes_are_said(self):
        bad = [{"split_serve": 1}, {"split_serve": {"prefill_gpu": 0}}, {"split_serve": {"prefill_gpu": 1, "decode_gpu": 1}},
               {"split_serve": {"prefill_gpu": 0, "decode_gpu": -1}}, {"split_serve": {"prefill_gpu": True, "decode_gpu": 1}},
               {"split_serve": {"prefill_gpu": 0, "decode_gpu": 1, "x": 1}},
               {"split_serve": {"prefill_gpu": 0, "decode_gpu": 1, "min_prompt_tokens": "9"}},
               {"split_serve": {"prefill_gpu": 0, "decode_gpu": 1, "prefill_args": "--x"}},
               {"split_serve": {"prefill_gpu": 0, "decode_gpu": 1, "shared_arena": 3}}]
        for cfg in bad:
            with self.assertRaises(SplitConfigError, msg=cfg):
                split_spec(cfg)

    def test_check_cfg_refuses_what_cannot_run(self):
        spec = split_spec({"split_serve": {"prefill_gpu": 0, "decode_gpu": 1}})
        check_cfg({"args": ["--serve", "--prompt-cache", "3"]}, spec)
        for cfg in ({"args": ["--layer-split", "24"]}, {"args": ["--batch", "2"]}, {"args": ["--prompt-cache", "0"]},
                    {"args": [], "replicas": 2}, {"args": [], "parallel": 2}, {"args": ["--peer-device", "1"]}):
            with self.assertRaises(SplitConfigError, msg=cfg):
                check_cfg(cfg, spec)
        spec2 = split_spec({"split_serve": {"prefill_gpu": 0, "decode_gpu": 1, "decode_args": ["--batch", "2"]}})
        with self.assertRaises(SplitConfigError):
            check_cfg({"args": []}, spec2)

    def test_role_cfg(self):
        cfg = {"gpu": [0, 1], "args": ["--x"], "log": "/l/strata.log", "hip_ordinal": 1, "gpu_roles": {"0": "main"},
               "expert_profile_save": "/p/prof.bin",
               "split_serve": {"prefill_gpu": 2, "decode_gpu": 3, "decode_args": ["--expert-cache", "auto"]}}
        spec = split_spec(cfg)
        p, d = role_cfg(cfg, spec, "prefill"), role_cfg(cfg, spec, "decode")
        self.assertEqual((p["gpu"], d["gpu"]), (2, 3))
        self.assertEqual((p["log"], d["log"]), ("/l/strata.prefill.log", "/l/strata.decode.log"))
        self.assertEqual(p["args"], ["--x"])
        self.assertEqual(d["args"], ["--x", "--expert-cache", "auto"])
        self.assertEqual(d["expert_profile_save"], "/p/prof.decode.bin")
        for c in (p, d):
            self.assertNotIn("split_serve", c)
            self.assertNotIn("hip_ordinal", c)
            self.assertNotIn("gpu_roles", c)
        self.assertEqual(cfg["args"], ["--x"])                       # the original is untouched

    def test_arena_choice(self):
        spec = split_spec({"split_serve": {"prefill_gpu": 0, "decode_gpu": 1}})
        off = split_spec({"split_serve": {"prefill_gpu": 0, "decode_gpu": 1, "shared_arena": False}})
        named = split_spec({"split_serve": {"prefill_gpu": 0, "decode_gpu": 1, "shared_arena": "/data/a.bin"}})
        self.assertIsNone(sp.arena_path({"args": ["--shared-expert-arena", "/x"]}, spec)[0])
        self.assertIsNone(sp.arena_path({"args": []}, off)[0])
        if os.name != "nt":
            self.assertEqual(sp.arena_path({"args": []}, named)[0], os.path.abspath("/data/a.bin"))
            path, note = sp.arena_path({"args": ["--pack", "/no/such/pack"]}, spec)      # nothing to size it from
            self.assertIsNone(path)
            self.assertIn("experts.bin", note)

    def test_common_prefix(self):
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 9, 4]), 2)
        self.assertEqual(common_prefix([], [1]), 0)
        self.assertEqual(common_prefix([1, 2], [1, 2]), 2)


class FakeEngine:
    """The part of StrataEngine the SplitEngine uses.  `session` is the token list its session holds (a file is a json-free
    list in a dict keyed by path, shared between the two fakes)."""
    files = {}

    def __init__(self, name, tokens_per_answer=3):
        self.name, self.up, self.max_context, self.info, self.last = name, True, 4096, {}, {}
        self.session, self.log, self.progress, self.prefill_tok_s_mean = [], [], None, None
        self.answer = tokens_per_answer
        self.fail_save = self.fail_restore = self.fail_prefill = None
        self.unloaded = False

    def alive(self):
        return self.up

    def restart(self, tries=3):
        self.up = True

    def exit_code(self):
        return None

    def death_note(self):
        return ""

    def unload(self):
        self.up = False

    def close(self):
        self.up = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False):
        if not self.up:
            raise EngineDied("down")
        if prefill_only:
            if self.fail_prefill:
                raise self.fail_prefill
            have = common_prefix(self.session, ids[:-1])
            self.log.append(("prefill", len(ids)))
            self.session = list(ids[:-1])
            self.last = {"generated": 0, "prompt_tokens": len(ids), "prompt_ms": 1000.0, "decode_ms": 0.0,
                         "finish": "length", "reused": have, "prompt_read": len(ids) - have}
            yield None
            return
        have = common_prefix(self.session, ids[:-1])
        out = [100 + i for i in range(min(max_new, self.answer))]
        self.log.append(("generate", len(ids), have))
        self.session = list(ids) + out[:-1]
        self.last = {"generated": len(out), "prompt_tokens": len(ids), "prompt_ms": 50.0, "decode_ms": 200.0,
                     "finish": "length", "reused": have, "prompt_read": len(ids) - have}
        for t in out:
            yield t

    def session_file(self, action, path):
        if action == "save":
            if self.fail_save:
                raise self.fail_save
            FakeEngine.files[path] = list(self.session)
            with open(path, "wb") as f:
                f.write(b"x" * 1000)
            return {"tokens": len(self.session), "bytes": 1000, "ms": 12.0}
        if self.fail_restore:
            raise self.fail_restore
        self.session = list(FakeEngine.files[path])
        return {"tokens": len(self.session), "bytes": 1000, "ms": 7.0}


class Handover(unittest.TestCase):
    def setUp(self):
        FakeEngine.files = {}
        self.dir = tempfile.mkdtemp(prefix="split-test-")
        self.p, self.d = FakeEngine("p"), FakeEngine("d")
        self.said = []
        self.e = SplitEngine(self.p, self.d, min_tokens=100, work_dir=self.dir, say=self.said.append)
        self.cancel = threading.Event()

    def run_one(self, ids, max_new=5):
        return [t for t in self.e.generate(ids, max_new, {}, self.cancel) if t is not None]

    def test_a_long_prompt_is_handed_over(self):
        ids = list(range(500))
        out = self.run_one(ids)
        self.assertEqual(out, [100, 101, 102])
        self.assertEqual(self.p.log, [("prefill", 500)])
        self.assertEqual(self.d.log, [("generate", 500, 499)])         # the decode engine read one token itself
        self.assertEqual(self.e.stats["handed_over"], 1)
        self.assertEqual(os.listdir(self.dir), [])                      # the session file is gone
        last = self.e.last
        self.assertEqual(last["reused"], 0)
        self.assertEqual(last["prompt_read"], 500)                     # 499 read by the prefill engine, 1 by the decode one... 500
        self.assertGreaterEqual(last["prompt_ms"], 12.0 + 7.0 + 50.0)                  # the file steps and the decode engine's start
        self.assertLess(last["prompt_ms"], 12.0 + 7.0 + 50.0 + 1000.0)
        self.assertEqual(last["split_bytes"], 1000)

    def test_a_short_prompt_stays_on_the_decode_engine(self):
        self.run_one(list(range(50)))
        self.assertEqual(self.p.log, [])
        self.assertEqual(self.e.stats["short"], 1)

    def test_the_next_turn_is_not_handed_over_when_the_decode_engine_holds_it(self):
        ids = list(range(500))
        out = self.run_one(ids)
        turn2 = ids + out + list(range(7000, 7040))                      # the answer and a short question
        self.run_one(turn2)
        self.assertEqual(len(self.p.log), 1)                            # only the first turn went through the prefill engine
        self.assertEqual(self.d.log[-1][0], "generate")
        self.assertGreater(self.d.log[-1][2], 490)                      # and the decode engine continued its own session

    def test_a_big_new_part_is_handed_over_again(self):
        ids = list(range(500))
        out = self.run_one(ids)
        self.run_one(ids + out + list(range(9000, 9400)))
        self.assertEqual(len(self.p.log), 2)

    def test_pictures_go_to_the_decode_engine(self):
        list(self.e.generate(list(range(500)), 3, {}, self.cancel, embeddings="/tmp/x"))
        self.assertEqual(self.p.log, [])

    def test_a_dead_prefill_engine_is_not_an_error(self):
        self.p.up = False
        self.e.heal = lambda: None
        out = self.run_one(list(range(500)))
        self.assertEqual(out, [100, 101, 102])
        self.assertEqual(self.d.log, [("generate", 500, 0)])            # the decode engine read all of it

    def test_a_prefill_engine_that_dies_midway_falls_back(self):
        self.p.fail_prefill = EngineDied("gone")
        self.e.heal = lambda: None
        out = self.run_one(list(range(500)))
        self.assertEqual(out, [100, 101, 102])
        self.assertEqual(self.d.log, [("generate", 500, 0)])
        self.assertTrue(any("failed" in s for s in self.said))

    def test_a_refused_file_falls_back(self):
        for which in ("fail_save", "fail_restore"):
            FakeEngine.files = {}
            self.p.session, self.d.session = [], []
            setattr(self.p if which == "fail_save" else self.d, which, SessionRefused("invalid", "other model", False))
            self.d.log.clear()
            ids = list(range(500)) if which == "fail_save" else list(range(1000, 1500))
            out = self.run_one(ids)
            self.assertEqual(out, [100, 101, 102])
            self.assertEqual(self.d.log, [("generate", 500, 0)], which)
            setattr(self.p if which == "fail_save" else self.d, which, None)
        self.assertEqual(self.e.stats["fallbacks"], 2)
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_decode_engine_that_dies_in_the_restore_is_an_error(self):
        def boom(*a):
            self.d.up = False
            raise EngineDied("transfer failed")
        self.d.fail_restore = None
        orig = self.d.session_file
        self.d.session_file = lambda action, path: boom() if action == "restore" else orig(action, path)
        with self.assertRaises(EngineDied):
            self.run_one(list(range(500)))
        self.assertEqual(os.listdir(self.dir), [])

    def test_cancel_during_the_prefill_ends_quietly(self):
        self.cancel.set()
        orig = self.p.generate

        def gen(*a, **k):
            for t in orig(*a, **k):
                yield t
        self.p.generate = gen
        out = self.run_one(list(range(500)))
        self.assertEqual(out, [])
        self.assertEqual(self.d.log, [])
        self.assertEqual(self.e.held, [])

    def test_attributes_come_from_the_decode_engine(self):
        self.assertEqual(self.e.max_context, 4096)
        self.assertEqual(self.e.batch, 0)
        self.assertFalse(hasattr(self.e, "session_file"))
        self.assertTrue(self.e.alive())
        self.d.up = False
        self.assertFalse(self.e.alive())
        self.assertTrue(self.e.info["split_serve"])
        self.e.silence_s = 5
        self.assertEqual((self.p.silence_s, self.d.silence_s), (5, 5))

    def test_close_removes_the_work_dir(self):
        self.e.close()
        self.assertFalse(os.path.exists(self.dir))
        self.assertFalse(self.p.up or self.d.up)


@unittest.skipUnless(os.name != "nt" and os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK),
                     "private executable-child/generated-arena profile requires writable Linux /dev/shm")
class StartupOwnerRetirement(unittest.TestCase):
    @contextlib.contextmanager
    def pair(self, mode="healthy"):
        import serve.server as server
        # Local class import avoids collecting a second copy of its test cases.
        from serve.test_ready_timeout import ReadyTimeout
        directory = tempfile.mkdtemp(prefix="split-owner-test-")
        cleanup_complete = False
        try:
            root = Path(directory)
            work = root / "work"
            fd, arena_name = tempfile.mkstemp(prefix="strata-split-", dir="/dev/shm")
            os.write(fd, b"private generated-arena recording backing")
            os.close(fd)
            arena = Path(arena_name)
            script = root / "engine.py"
            script.write_text("#!" + sys.executable + "\nimport sys\n"
                + ("if sys.argv[sys.argv.index('--role')+1]=='decode': raise SystemExit(1)\n" if mode == "second_exit" else "")
                + "print('READY 4096 stop', flush=True)\n"
                + "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
            script.chmod(0o700)
            cfg = {"args": [], "split_serve": {"prefill_gpu": 0, "decode_gpu": 1,
                   "dir": str(work), "shared_arena": True,
                   "prefill_args": ["--role", "prefill"], "decode_args": ["--role", "decode"]}}
            spec = sp.split_spec(cfg)
            state = SimpleNamespace(server=server, cfg=cfg, spec=spec, script=script, work=work,
                                    arena=arena, engines=[], adapters=[], prevented=[])
            with mock.patch.dict(os.environ, {"STRATA_KEEP_LEFTOVERS": "1"}), \
                 mock.patch.object(sp, "arena_path", return_value=(str(arena), "private recording backing")), \
                 mock.patch.object(server, "engine_args", side_effect=lambda cfg, *a, **k: cfg.get("args", [])), \
                 mock.patch.object(server, "child_env", return_value={}), \
                 mock.patch.object(server, "mtp_retry_args", return_value=None), \
                 mock.patch.object(server, "nodraft_retry_args", return_value=None), \
                 mock.patch.object(server, "warn_budget_over_ram"), \
                 ReadyTimeout.captured_start(self, str(script)) as (_, children, readers, _):
                state.children, state.readers = children, readers
                try:
                    yield state
                finally:
                    # Explicitly end/wait the real private children BEFORE
                    # resolving only the starts our deliberate injector prevented.
                    for proc, wait, kill, _, _, _ in children:
                        if proc.poll() is None:
                            kill()
                        wait(timeout=5)
                    for reader, original_start in state.prevented:
                        original_start(reader)
                        reader.join(5)
                        self.assertFalse(reader.is_alive())
                    for engine in state.engines:
                        engine.close()
                    for adapter in state.adapters:
                        adapter.close()
            cleanup_complete = True
        finally:
            if cleanup_complete:
                # Never remove fixture-owned backing after an unresolved cleanup.
                if arena.exists():
                    arena.unlink()
                shutil.rmtree(directory, ignore_errors=True)

    def start_pair(self, state):
        return state.server.start_split(state.cfg, state.spec, str(state.script), [], False)

    def test_second_role_start_failure_retains_concrete_roles_and_borrowed_paths(self):
        with self.pair() as state:
            cls = type(threading.Thread(target=lambda: None))
            original_start = cls.start
            def start(thread):
                target = getattr(thread, "_target", None)
                if target == state.server.StrataEngine._ready_pump:
                    proc = thread._args[0]
                    if proc.args[proc.args.index("--role") + 1] == "decode":
                        state.prevented.append((thread, original_start))
                        raise RuntimeError("deliberately prevented actual decode READY start")
                return original_start(thread)
            with mock.patch.object(cls, "start", start):
                with self.assertRaises(state.server.EngineStuck) as caught:
                    self.start_pair(state)
            error = caught.exception
            owners = error.startup_engines
            state.engines.extend(owners.values())
            self.assertEqual(set(owners), {"prefill", "decode"})
            self.assertEqual(len(state.children), 2)
            self.assertIs(owners["prefill"]._epoch.proc, state.children[0][0])
            self.assertIs(owners["decode"].proc, state.children[1][0])
            self.assertIs(owners["decode"]._startup_attempt.ready_reader, state.prevented[0][0])
            self.assertIs(error.startup_failure.startup_engine, owners["decode"])
            self.assertIsInstance(error.__cause__, state.server.EngineStuck)
            self.assertEqual(error.startup_work_dir, str(state.work))
            self.assertEqual(error.startup_arena, str(state.arena))
            self.assertTrue(state.work.is_dir())
            self.assertTrue(state.arena.is_file())
            self.assertFalse(owners["decode"]._startup_attempt.retired)

    def test_split_close_uncertainty_retains_adapter_roles_work_and_arena(self):
        with self.pair() as state:
            adapter = self.start_pair(state)
            state.adapters.append(adapter)
            first = adapter.prefill
            first.wlock.acquire()
            try:
                with mock.patch.object(state.server, "ENGINE_QUIT_WAIT_S", .05), \
                     mock.patch.object(state.server, "ENGINE_TERM_WAIT_S", .1), \
                     mock.patch.object(state.server, "ENGINE_KILL_WAIT_S", .1):
                    with self.assertRaises(state.server.EngineStuck) as caught:
                        adapter.close()
                error = caught.exception
                self.assertIs(error.startup_adapter, adapter)
                self.assertEqual(error.startup_engines, (adapter.prefill, adapter.decode))
                self.assertIsInstance(error.__cause__, state.server.EngineStuck)
                self.assertIsNotNone(first.__dict__.get("_close_owner"))
                self.assertTrue(state.work.is_dir())
                self.assertTrue(state.arena.is_file())
                self.assertEqual(len(state.children), 2)
            finally:
                first.wlock.release()
                adapter.close()
            self.assertFalse(state.work.exists())
            self.assertFalse(state.arena.exists())
            self.assertTrue(all(proc.poll() is not None and stdin.closed and stdout.closed
                                for proc, _, _, stdin, stdout, _ in state.children))

    def test_healthy_split_close_retires_both_children_before_generated_paths(self):
        with self.pair() as state:
            adapter = self.start_pair(state)
            state.adapters.append(adapter)
            self.assertEqual(len(state.children), 2)
            self.assertIs(adapter.prefill.proc, state.children[0][0])
            self.assertIs(adapter.decode.proc, state.children[1][0])
            self.assertTrue(state.work.is_dir())
            self.assertTrue(state.arena.is_file())
            adapter.close()
            self.assertTrue(all(proc.poll() is not None and stdin.closed and stdout.closed
                                for proc, _, _, stdin, stdout, _ in state.children))
            self.assertIsNone(adapter.prefill.proc)
            self.assertIsNone(adapter.decode.proc)
            self.assertFalse(state.work.exists())
            self.assertFalse(state.arena.exists())

    def test_fully_retired_second_role_failure_preserves_original_engine_died(self):
        with self.pair("second_exit") as state:
            with self.assertRaises(state.server.EngineDied) as caught:
                self.start_pair(state)
            state.engines.append(caught.exception.startup_engine)
            self.assertEqual(len(state.children), 2)
            self.assertTrue(all(proc.poll() is not None and stdin.closed and stdout.closed
                                for proc, _, _, stdin, stdout, _ in state.children))
            self.assertFalse(state.work.exists())
            # Preserve original startup-failure arena policy; adapter close has
            # the separate generated-arena cleanup contract above.
            self.assertTrue(state.arena.exists())


if __name__ == "__main__":
    unittest.main()

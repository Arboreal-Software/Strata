"""Data-parallel replicas (serve/replicas.py): the config, the router, and the real Service over HTTP with two fake
engines (the fake of test_parallel.py) behind a ReplicaEngine."""
import json
import contextlib
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.replicas import (BLOCK, ReplicaConfigError, ReplicaEngine, Router, cpu_sets, gpu_groups, remove_arena,
                            replica_cfg, shared_arena_plan, sweep_stale_arenas)
from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.test_parallel import FAKE_BATCH


class Config(unittest.TestCase):
    def test_a_number_cuts_the_gpu_list(self):
        self.assertEqual(gpu_groups({"gpu": [0, 1, 2, 3], "replicas": 2}), [[0, 1], [2, 3]])
        self.assertEqual(gpu_groups({"gpu": "0,1,2,3", "replicas": 4}), [[0], [1], [2], [3]])
        self.assertEqual(gpu_groups({"gpu": [0, 1], "replicas": 1}), [])
        self.assertEqual(gpu_groups({"gpu": [0, 1]}), [])

    def test_named_groups(self):
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}, {"gpus": [2]}]}), [[0, 1], [2]])
        self.assertEqual(gpu_groups({"replicas": [{"gpus": [0, 1]}]}), [])

    def test_mistakes_are_said(self):
        for cfg in ({"gpu": [0, 1, 2], "replicas": 2}, {"replicas": 2}, {"replicas": "2"},
                    {"replicas": [{"gpus": [0, 1]}, {"gpus": [1, 2]}]}, {"replicas": [{"gpus": []}, {"gpus": [1]}]}):
            with self.assertRaises(ReplicaConfigError, msg=cfg):
                gpu_groups(cfg)

    def test_replica_cfg(self):
        cfg = {"gpu": [0, 1, 2, 3], "args": ["--x"], "log": "/l/strata.log", "hip_ordinal": 1, "env": {"A": "1"},
               "expert_profile_save": "/p/prof.bin",
               "replicas": [{"gpus": [0, 1], "env": {"B": 2}}, {"gpus": [2, 3], "args": ["--y"]}]}
        a, b = replica_cfg(cfg, 0, [0, 1]), replica_cfg(cfg, 1, [2, 3])
        self.assertEqual((a["gpu"], b["gpu"]), ([0, 1], [2, 3]))
        self.assertEqual((a["log"], b["log"]), ("/l/strata.r0.log", "/l/strata.r1.log"))
        self.assertEqual(a["env"], {"A": "1", "B": "2"})
        self.assertEqual(b["args"], ["--x", "--y"])
        self.assertNotIn("replicas", a)
        self.assertNotIn("hip_ordinal", a)
        self.assertEqual(a["expert_profile_save"], "/p/prof.r0.bin")
        self.assertEqual(cfg["args"], ["--x"])                       # the original is untouched
        self.assertEqual(replica_cfg({"gpu": [0, 1], "args": []}, 0, [3])["gpu"], 3)

    def test_cpu_sets_off(self):
        self.assertIsNone(cpu_sets(2, enabled=False))
        self.assertIsNone(cpu_sets(1))


class Routing(unittest.TestCase):
    def test_matched_prefix_per_replica(self):
        r = Router(3)
        a = list(range(BLOCK * 4))
        r.note(0, a)
        r.note(1, a[:BLOCK * 2])
        self.assertEqual(r.matched(a + [9] * 70), [BLOCK * 4, BLOCK * 2, 0])
        self.assertEqual(r.matched([7] * 200), [0, 0, 0])
        r.forget(0)
        self.assertEqual(r.matched(a), [0, BLOCK * 2, 0])

    def test_the_index_is_bounded(self):
        r = Router(2, cap=10)
        for i in range(50):
            r.note(i % 2, [i] * (BLOCK * 2))
        self.assertLessEqual(len(r.blocks), 10)


class FakeEngine:
    def __init__(self, n=4):
        self.batch, self.up = n, True
        self.max_context = 4096
        self.info = {}
        self.last = {}

    def alive(self):
        return self.up

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        yield 7


class RouteChoice(unittest.TestCase):
    def test_least_loaded_then_affinity(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        picks = [e.route([i] * 100) for i in range(4)]             # nothing cached: spread
        self.assertEqual(sorted(picks), [0, 0, 1, 1])
        ids = list(range(BLOCK * 8))
        e.inflight = [0, 0]
        k = e.route(ids)
        e.router.note(k, ids)
        e.inflight = [0, 0]
        self.assertEqual(e.route(ids + [5] * 10), k)                # the conversation goes back
        e.inflight = [0, 0]
        e.inflight[k] = 1                                           # 512 tokens held < 3000 per running request
        self.assertNotEqual(e.route(ids + [5] * 10), k)

    def test_a_long_prefix_outweighs_load(self):
        e = ReplicaEngine([FakeEngine(), FakeEngine()], load_tokens=3000)
        ids = list(range(BLOCK * 200))                              # 12,800 tokens held on replica 0
        e.router.note(0, ids)
        e.inflight = [2, 0]
        self.assertEqual(e.route(ids + [1]), 0)
        e.inflight = [5, 0]
        self.assertEqual(e.route(ids + [1]), 1)

    def test_a_dead_replica_is_skipped(self):
        a, b = FakeEngine(), FakeEngine()
        e = ReplicaEngine([a, b])
        e.heal = lambda: None
        ids = list(range(BLOCK * 3))
        e.router.note(0, ids)
        a.up = False
        self.assertEqual(e.route(ids), 1)
        b.up = False
        self.assertEqual(e.route(ids), -1)

    def test_nonfinite_forgets_the_replica(self):
        """#879: after DONE ... nonfinite the engine holds no conversation: the router forgets that replica."""
        from serve.server import EngineNonFinite

        class NonFinite(FakeEngine):
            def generate(self, ids, max_new, sampling, cancel):
                yield 7
                raise EngineNonFinite("the engine's logits became non-finite (#879)")

        e = ReplicaEngine([NonFinite(), NonFinite()])
        other, ids = [9] * (BLOCK * 4), list(range(BLOCK * 3))
        e.router.note(0, other)
        e.router.note(1, other)
        k = 0
        e.inflight[k] += 1                                          # as route() counts it
        with mock.patch.object(e, "route", return_value=k):
            with self.assertRaises(EngineNonFinite):
                list(e.generate(ids, 10, {}, threading.Event()))
        self.assertEqual(e.router.matched(ids)[k], 0)
        self.assertEqual(e.router.matched(other)[k], 0)             # its older conversations too
        self.assertEqual(e.router.matched(other)[1 - k], len(other))
        self.assertEqual(e.inflight[k], 0)

    def test_batch_is_the_sum(self):
        self.assertEqual(ReplicaEngine([FakeEngine(4), FakeEngine(4)]).batch, 8)
        self.assertEqual(ReplicaEngine([FakeEngine(0), FakeEngine(0)]).batch, 2)

    def test_selected_replica_is_reported_for_solo_requests_too(self):
        engine = ReplicaEngine([FakeEngine(), FakeEngine()])
        engine.inflight[0] = 1
        cancel = threading.Event()
        selected = []
        cancel._strata_replica_selected = selected.append
        self.assertEqual(list(engine.generate([1, 2], 1, {}, cancel)), [7])
        self.assertEqual(selected, [1])


class ReplicaService(unittest.TestCase):
    def start(self, slots=4, n=2):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_BATCH, encoding="utf-8")
        real = server.subprocess.Popen
        self.logs = [Path(self.tmp.name) / f"r{i}.log" for i in range(n)]
        engines = []
        for i in range(n):
            extra = (["--batch", str(slots)] if slots else []) + ["--log", str(self.logs[i])]
            with mock.patch.object(server.subprocess, "Popen",
                                   lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw)):
                engines.append(StrataEngine("strata", extra))
        self.engine = ReplicaEngine(engines)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        if getattr(self, "httpd", None):
            self.httpd.shutdown()
            self.httpd.server_close()
        if getattr(self, "engine", None):
            self.engine.unload()
        if getattr(self, "tmp", None):
            self.tmp.cleanup()

    def post(self, messages, max_tokens=64):
        body = {"messages": messages, "max_tokens": max_tokens, "reasoning_effort": "none"}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"]

    def chat(self, text, max_tokens=64):
        return self.post([{"role": "user", "content": text}], max_tokens)

    def served(self):
        return [sum(1 for l in p.read_text().splitlines() if l.split()[0] in ("GEN", "BGEN")) if p.exists() else 0
                for p in self.logs]

    def burst(self, prompts):
        out = [None] * len(prompts)

        def one(i):
            out[i] = self.chat(prompts[i])
        ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    def test_requests_spread_over_both_and_answers_are_the_solo_answer(self):
        self.start()
        solo = self.chat("hello there")
        before = self.served()
        out = self.burst([f"request number {i} " + "x" * 300 for i in range(6)])
        got = [b - a for a, b in zip(before, self.served())]
        self.assertTrue(all(g >= 2 for g in got), got)
        self.assertEqual(set(out), {solo})

    def test_metrics_and_status_see_all_slots(self):
        self.start()
        self.assertEqual(self.svc.v1_status()["concurrency"]["serving"], 8)
        m = self.svc.metrics()
        self.assertEqual(len(m["live"]["slots"]), 8)
        self.assertEqual([r["replica"] for r in m["live"]["replicas"]], [0, 1])
        self.assertEqual(m["engine"]["replicas"], 2)

    def test_terminal_telemetry_correlates_second_replica_without_global_sample_claim(self):
        self.start()
        self.svc.replica_gpu_groups = [[0, 1], [2, 3]]
        second = self.engine.engines[1]
        request_id = "abcdef012345"
        second.slot_live[1] = {"request_id": request_id, "state": "reading", "prompt_tokens": 100,
                               "prompt_read": 12, "generated": 0, "started": time.time(), "first_token": None}
        first = self.engine.engines[0]
        first.terminal_batch_sample = {"windows": 2, "processed_rows": 8, "active_rows": 4,
                                       "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        second.terminal_batch_sample = {"windows": 2, "processed_rows": 6, "active_rows": 4,
                                        "elapsed_ms": 1000, "observed_monotonic": time.monotonic()}
        first.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [800_000, 600_000],
                                       "observed_monotonic": time.monotonic()}
        second.terminal_stage_sample = {"elapsed_us": 1_000_000, "in_flight_us": [700_000, 500_000],
                                        "observed_monotonic": time.monotonic()}
        self.svc.backend = "cuda"
        for child, cards in ((first, [0, 1]), (second, [2, 3])):
            child.spawn = (*child.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                              "CUDA_VISIBLE_DEVICES": ",".join(map(str, cards))},
                           *child.spawn[5:])
            child.terminal_stages = [
                {"stage": 0, "layer_start": 0, "layer_end": 19, "engine_ordinal": 0},
                {"stage": 1, "layer_start": 20, "layer_end": 39, "engine_ordinal": 1}]
        with self.svc.status_lock:
            self.svc.terminal_requests[request_id] = {"id": request_id, "state": "waiting",
                                                       "queued_at": time.time(), "admitted_at": time.time(),
                                                       "prompt_tokens": 100}
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            snapshot = json.load(response)
        slot = snapshot["work"]["slots"][5]
        self.assertEqual((slot["slot"], slot["replica"], slot["group"], slot["request_id"]),
                         (5, 1, 1, request_id))
        request = snapshot["work"]["requests"][0]
        self.assertEqual((request["slot"], request["replica"], request["group"], request["state"]),
                         (5, 1, 1, "reading"))
        self.assertEqual(snapshot["work"]["groups"], 2)
        self.assertIsNone(snapshot["work"]["effective_rows_window"])
        self.assertEqual([(row["replica"], row["batch_sample_rows"], row["batch_sample_active_rows"])
                          for row in snapshot["work"]["replica_batches"]], [(0, 8, 4), (1, 6, 4)])
        self.assertEqual([(row["replica"], row["stage_in_flight_us"])
                          for row in snapshot["work"]["replica_stage_samples"]],
                         [(0, [800_000, 600_000]), (1, [700_000, 500_000])])
        self.assertNotIn("stage_in_flight_us", snapshot["work"])
        self.assertEqual(snapshot["topology"]["stages"], [])
        self.assertEqual(snapshot["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)
        self.assertEqual(snapshot["topology"]["replica_devices"],
                         [{"replica": 0, "devices": [0, 1]}, {"replica": 1, "devices": [2, 3]}])
        second.spawn = (*second.spawn[:4], {"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                            "CUDA_VISIBLE_DEVICES": "9,3"}, *second.spawn[5:])
        with urllib.request.urlopen(self.base + "/v1/telemetry", timeout=10) as response:
            mismatch = json.load(response)
        self.assertIsNone(mismatch["topology"]["replica_stages"][1]["stages"][0]["device_index"])
        self.assertEqual(mismatch["topology"]["replica_stages"][1]["stages"][1]["device_index"], 3)

    def test_engines_without_slots_run_one_each_at_once(self):
        self.start(slots=0)
        out = self.burst(["a" * 40, "b" * 40])
        self.assertEqual(self.served(), [1, 1])
        self.assertEqual(len(set(out)), 1)

    def test_a_conversation_goes_back_to_its_replica(self):
        self.start()
        first = "begin " + "lorem ipsum " * 40
        reply = self.chat(first)
        k = self.served().index(1)
        self.post([{"role": "user", "content": first}, {"role": "assistant", "content": reply},
                   {"role": "user", "content": "and then?"}], 16)
        self.assertEqual(self.served()[k], 2)

    def test_a_replica_that_died_is_skipped(self):
        self.start()
        self.engine.heal = lambda: None
        self.engine.engines[0].proc.kill()
        self.engine.engines[0].proc.wait()
        self.assertTrue(self.engine.alive())
        for i in range(3):
            self.assertIn("ok, done.", self.chat(f"after the loss {i}"))
        self.assertEqual(self.served()[1], 3)


class SharedArena(unittest.TestCase):
    """The opt-in shared expert arena of the replicas (shared_arena_plan): when it is used, when it is not, what a stale
    file means."""
    CFG = {"replica_shared_arena": True, "args": ["--pack", "/p"], "gpu": [0, 1]}

    def plan(self, cfg=None, n=2, tmp=None, **kw):
        return shared_arena_plan(cfg or self.CFG, n, shm=tmp, platform=kw.pop("platform", "linux"),
                                 environ=kw.pop("environ", {}), free_bytes=kw.pop("free_bytes", 100 << 30),
                                 use_memfd=kw.pop("use_memfd", False), **kw)

    def test_off_by_default_and_for_one_engine(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self.plan({"args": []}, tmp=d), (None, ""))
            self.assertEqual(self.plan(n=1, tmp=d), (None, ""))
            self.assertEqual(self.plan({"args": []}, tmp=d, environ={"STRATA_REPLICA_SHARED_ARENA": "0"}), (None, ""))

    def test_on_by_config_or_environment(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d)
            self.assertTrue(path and path.startswith(d) and path.endswith(".bin"), path)
            self.assertIn("once", note)
            path2, _ = self.plan({"args": []}, tmp=d, environ={"STRATA_REPLICA_SHARED_ARENA": "1"})
            self.assertTrue(path2)
            self.assertEqual(path, path2)      # (one server: one file)

    def test_windows_gets_a_named_section(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d, platform="win32")
            self.assertTrue(path and path.startswith("win:Local") and "strata-replicas-" in path, path)
            self.assertIn("Windows section", note)
            remove_arena(path)                 # (nothing to remove: harmless)

    def test_only_where_there_is_an_arena_to_share(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self.plan(tmp=d, platform="darwin")[0])
            for flag in ("--mmap-experts", "--resident-experts", "--resident-budget-gib", "--shared-expert-arena"):
                self.assertIsNone(self.plan({**self.CFG, "args": ["--pack", "/p", flag]}, tmp=d)[0], flag)
            self.assertIsNone(self.plan({**self.CFG, "replicas": [{"gpus": [0], "args": ["--mmap-experts"]}, {"gpus": [1]}]}, tmp=d)[0])
            self.assertIsNone(self.plan({**self.CFG, "arch": "deepseek41"}, tmp=d)[0])

    def test_no_room_or_no_shm_says_so_and_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            path, note = self.plan(tmp=d, free_bytes=2 << 30)
            self.assertIsNone(path)
            self.assertIn("2.0 GiB free", note)
            path, note = self.plan(tmp=d + "/nope")
            self.assertIsNone(path)
            self.assertIn("not usable", note)

    def test_a_stale_file_goes_a_live_one_stays(self):
        with tempfile.TemporaryDirectory() as d:
            uid = __import__("os").getuid() if hasattr(__import__("os"), "getuid") else 0
            dead = Path(d) / f"strata-replicas-{uid}-2999999.bin"
            live = Path(d) / f"strata-replicas-{uid}-{__import__('os').getpid()}.bin"
            other = Path(d) / f"strata-replicas-{uid + 1}-2999999.bin"
            junk = Path(d) / "strata-replicas-x.bin"
            for f in (dead, live, other, junk):
                f.write_bytes(b"x")
            with mock.patch("serve.replicas._pid_alive", side_effect=lambda pid: pid != 2999999):
                gone = sweep_stale_arenas(d, uid)
            self.assertEqual(gone, [dead.name])
            self.assertFalse(dead.exists())
            self.assertTrue(live.exists() and other.exists() and junk.exists())

    @unittest.skipUnless(hasattr(__import__("os"), "memfd_create") and sys.platform.startswith("linux"), "Linux memfd")
    def test_a_memfd_is_not_limited_by_dev_shm_and_is_closed_at_the_end(self):
        import os
        path, note = shared_arena_plan(self.CFG, 2, platform="linux", environ={}, free_bytes=100 << 30, use_memfd=True)
        self.assertTrue(path and path.startswith(f"/proc/{os.getpid()}/fd/"), path)
        fd = int(path.rsplit("/", 1)[1])
        os.ftruncate(fd, 1 << 20)
        self.assertEqual(os.fstat(fd).st_size, 1 << 20)
        with open(path, "r+b") as other:          # (how an engine opens it)
            other.seek(100)
            other.write(b"x")
        remove_arena(path)
        with self.assertRaises(OSError):
            os.fstat(fd)
        remove_arena(path)                         # (twice: harmless)

    @unittest.skipUnless(hasattr(__import__("os"), "memfd_create") and sys.platform.startswith("linux"), "Linux memfd")
    def test_a_memfd_arena_warns_when_the_engines_will_not_all_fit(self):
        cfg = {**self.CFG, "model_name": "qwen3.8-flash-next-iq3_s"}
        path, note = shared_arena_plan(cfg, 2, platform="linux", environ={}, free_bytes=57 << 30, use_memfd=True)
        try:
            self.assertTrue(path)
            self.assertIn("WARNING", note)
            self.assertIn("may not fit", note)
        finally:
            remove_arena(path)
        path, note = shared_arena_plan(cfg, 2, platform="linux", environ={}, free_bytes=120 << 30, use_memfd=True)
        try:
            self.assertNotIn("WARNING", note)
        finally:
            remove_arena(path)

    def test_a_memfd_needs_free_ram(self):
        path, note = shared_arena_plan(self.CFG, 2, platform="linux", environ={}, free_bytes=2 << 30, use_memfd=True)
        self.assertIsNone(path)
        self.assertIn("2.0 GiB of RAM", note)

    def test_removing_twice_or_nothing_is_harmless(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"x")
            remove_arena(str(f))
            remove_arena(str(f))
            remove_arena(None)
            self.assertFalse(f.exists())

    def test_the_replica_engine_unlinks_the_file_at_close(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"x")

            class E:
                def close(self):
                    pass
            r = ReplicaEngine([E(), E()])
            r.__dict__["arena_file"] = str(f)
            r.close()
            self.assertFalse(f.exists())


@unittest.skipIf(os.name == "nt", "the executable script uses the existing POSIX fake-engine boundary")
class StartupArenaRetirement(unittest.TestCase):
    @contextlib.contextmanager
    def group(self, mode):
        import serve.server as server
        from serve import replicas as rp
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arena = root / "arena"
            arena.write_bytes(b"actual parent-owned arena fixture")
            script = root / "engine.py"
            script.write_text("#!" + sys.executable + "\nimport sys,time\n"
                + ("if '--shared-expert-arena' in sys.argv: raise SystemExit(1)\n" if mode == "fallback" else "")
                + ("if sys.argv[sys.argv.index('--index')+1]=='0': raise SystemExit(1)\n" if mode == "cleanup" else "")
                + (("from pathlib import Path\nmarker=Path(" + repr(str(root / "first")) + ")\n"
                    "if '--index' in sys.argv and sys.argv[sys.argv.index('--index')+1]=='0' and not marker.exists():\n"
                    "    marker.touch()\n    raise SystemExit(1)\n") if mode == "same" else "")
                + "print('READY 4096 stop', flush=True)\n"
                + "for line in sys.stdin:\n    if line.startswith('QUIT'): break\n")
            script.chmod(0o700)
            children, readers, prevented = [], [], []
            real_popen, real_thread = server.popen, threading.Thread
            def popen(*args, **kwargs):
                proc = real_popen(*args, **kwargs)
                children.append(proc)
                return proc
            def thread(*args, **kwargs):
                result = real_thread(*args, **kwargs)
                if kwargs.get("target") == StrataEngine._ready_pump:
                    readers.append(result)
                    proc = kwargs["args"][0]
                    index = proc.args[proc.args.index("--index") + 1]
                    if mode == "mixed" and index == "0":
                        prevented.append((result, result.start))
                        result.start = mock.Mock(side_effect=RuntimeError("injected actual READY start failure"))
                return result
            cfg = {"replica_shared_arena": True, "args": [], "replica_cpus": False,
                   "replicas": [{"args": ["--index", "0"]}, {"args": ["--index", "1"]}]}
            remove = rp.remove_arena
            with mock.patch.dict(os.environ, {"STRATA_REPLICA_START": "serial"}), \
                 mock.patch.object(server, "popen", side_effect=popen), \
                 mock.patch.object(server.threading, "Thread", side_effect=thread), \
                 mock.patch.object(rp, "shared_arena_plan", side_effect=lambda cfg, n:
                                   (str(arena), "") if cfg.get("replica_shared_arena") else (None, "")), \
                 mock.patch.object(rp, "cpu_sets", return_value=None), \
                 mock.patch.object(server, "engine_args", side_effect=lambda cfg, *a, **k: cfg.get("args", [])), \
                 mock.patch.object(server, "child_env", return_value={}), \
                 mock.patch.object(server, "mtp_retry_args", return_value=None), \
                 mock.patch.object(server, "nodraft_retry_args", return_value=None), \
                 mock.patch.object(server, "warn_budget_over_ram"), \
                 mock.patch.object(rp, "remove_arena", wraps=remove) as removal:
                try:
                    yield server, cfg, script, arena, children, removal
                finally:
                    for proc in children:
                        if proc.poll() is None:
                            proc.kill()
                        proc.wait(timeout=5)
                    for reader, start in prevented:
                        reader.start = start
                        if reader.ident is None:
                            reader.start()
                    for reader in readers:
                        if reader.ident is not None:
                            reader.join(5)
                            self.assertFalse(reader.is_alive())
                    for proc in children:
                        proc.stdin.close()
                        proc.stdout.close()

    def test_array_owns_concrete_engine_before_constructor_entry(self):
        import serve.server as server
        engines = [None]
        original = StrataEngine.__init__
        entered = []
        def initialize(engine, *args, **kwargs):
            self.assertIs(engines[0], engine)
            entered.append(engine)
            return original(engine, *args, **kwargs)
        with mock.patch.object(StrataEngine, "__init__", initialize):
            result = server._construct_startup_engine("missing", [], lazy=True, _engines=engines, _index=0)
        self.assertEqual(entered, [result])
        self.assertIs(engines[0], result)
        self.assertIsNone(result.proc)
        result.close()

    def test_mixed_shared_arena_failure_retains_group_without_retry_or_unlink(self):
        with self.group("mixed") as (server, cfg, script, arena, children, removal):
            failed = None
            try:
                with self.assertRaises(server.EngineStuck) as caught:
                    server.start_replicas(cfg, [[0], [1]], str(script), [], False)
                failed = caught.exception
                self.assertEqual(len(children), 2)
                self.assertEqual(len(failed.startup_engines), 2)
                self.assertIs(failed.startup_engines[0].proc, children[0])
                self.assertIs(failed.startup_engines[1].proc, children[1])
                self.assertEqual(failed.startup_arena, str(arena))
                self.assertTrue(arena.exists())
                removal.assert_not_called()
            finally:
                # group() first resolves the real processes/readers; retain owners
                # until that context cleanup, rather than inventing retirement.
                if failed is not None:
                    self.addCleanup(self.retire_group, failed.startup_engines)

    def retire_group(self, engines):
        for engine in engines:
            if engine is not None:
                engine.close()

    def test_later_launcher_start_failure_retains_live_child_and_all_owners(self):
        with self.group("healthy") as (server, cfg, script, arena, children, removal):
            admitted, release = threading.Event(), threading.Event()
            actual_thread_class = type(threading.Thread(target=lambda: None))
            original_start, popen = actual_thread_class.start, server.popen
            attempted, started, prevented = [], [], []
            failed = None
            def spawn(*args, **kwargs):
                proc = popen(*args, **kwargs)
                admitted.set()  # A REAL child exists before the later start fault.
                if not release.wait(5):
                    raise AssertionError("child publication barrier not released")
                return proc
            def start(thread):
                if getattr(getattr(thread, "_target", None), "__name__", "") == "one":
                    attempted.append(thread)
                    if len(attempted) == 2:
                        self.assertTrue(admitted.wait(5))
                        prevented.append(thread)  # Deliberate fault BEFORE native start.
                        raise RuntimeError("deliberately prevented second launcher start")
                    result = original_start(thread)
                    started.append(thread)
                    return result
                return original_start(thread)
            try:
                with mock.patch.dict(os.environ, {"STRATA_REPLICA_START": ""}), \
                     mock.patch.object(actual_thread_class, "start", start), \
                     mock.patch.object(server, "popen", side_effect=spawn):
                    with self.assertRaises(server.EngineStuck) as caught:
                        server.start_replicas(cfg, [[0], [1]], str(script), [], False)
                    failed = caught.exception
                    self.assertIsInstance(failed.__cause__, RuntimeError)
                    self.assertEqual(len(children), 1)
                    self.assertIsNone(children[0].poll())
                    self.assertEqual(failed.startup_threads, attempted)
                    self.assertIs(failed.startup_threads[0], started[0])
                    self.assertIs(failed.startup_threads[1], prevented[0])
                    self.assertTrue(started[0].is_alive())
                    self.assertIsNotNone(failed.startup_engines[0])
                    self.assertTrue(failed.startup_engines[0]._startup_attempt.in_flight)
                    self.assertIsNone(failed.startup_engines[1])
                    self.assertEqual(failed.startup_arena, str(arena))
                    self.assertIsInstance(failed.startup_errors, list)
                    self.assertTrue(arena.exists())
                    removal.assert_not_called()
            finally:
                release.set()
                # Join only the launches we actually invoked. The other is known
                # absent from the deliberate injector, not from ident==None.
                for worker in started:
                    worker.join(5)
                    self.assertFalse(worker.is_alive())
                if failed is not None:
                    try:
                        self.assertIs(failed.startup_engines[0].proc, children[0])
                        self.assertEqual(len(children), 1)
                    finally:
                        self.retire_group(failed.startup_engines)
                    self.assertIsNotNone(children[0].poll())
                    self.assertTrue(children[0].stdin.closed)
                    self.assertTrue(children[0].stdout.closed)
                    removal.assert_not_called()

    def test_parallel_launcher_success_retires_children_before_arena_release(self):
        with self.group("healthy") as (server, cfg, script, arena, children, removal):
            with mock.patch.dict(os.environ, {"STRATA_REPLICA_START": ""}):
                group = server.start_replicas(cfg, [[0], [1]], str(script), [], False)
            try:
                self.assertEqual(len(children), 2)
                self.assertEqual({engine.proc for engine in group.engines}, set(children))
                self.assertTrue(all(proc.poll() is None for proc in children))
                self.assertTrue(arena.exists())
                removal.assert_not_called()
            finally:
                group.close()
            self.assertTrue(all(proc.poll() is not None for proc in children))
            self.assertTrue(all(proc.stdin.closed and proc.stdout.closed for proc in children))
            self.assertFalse(arena.exists())
            removal.assert_called_once_with(str(arena))

    def test_retired_failure_retries_same_arena_without_replacing_healthy_sibling(self):
        with self.group("same") as (server, cfg, script, arena, children, removal):
            group = server.start_replicas(cfg, [[0], [1]], str(script), [], False)
            try:
                self.assertEqual(len(children), 3)
                self.assertIsNotNone(children[0].poll())
                self.assertTrue(children[0].stdin.closed)
                self.assertTrue(children[0].stdout.closed)
                self.assertIs(group.engines[1].proc, children[1])
                self.assertIs(group.engines[0].proc, children[2])
                self.assertTrue(all("--shared-expert-arena" in proc.args for proc in children))
                self.assertTrue(arena.exists())
                removal.assert_not_called()
            finally:
                group.close()
            self.assertFalse(arena.exists())

    def test_fully_retired_arena_failure_allows_private_fallback(self):
        with self.group("fallback") as (server, cfg, script, arena, children, removal):
            group = server.start_replicas(cfg, [[0], [1]], str(script), [], False)
            try:
                self.assertEqual(len(children), 4)
                self.assertTrue(all(proc.poll() is not None for proc in children[:2]))
                self.assertTrue(all(proc.stdin.closed and proc.stdout.closed for proc in children[:2]))
                self.assertFalse(arena.exists())
                removal.assert_called_once_with(str(arena))
                self.assertEqual(len(group.engines), 2)
                self.assertTrue(all(engine.alive() for engine in group.engines))
            finally:
                group.close()

    def test_all_arena_fallback_stops_when_healthy_sibling_cannot_retire(self):
        with self.group("cleanup") as (server, cfg, script, arena, children, removal):
            construct = server._construct_startup_engine
            held, failed = [], None
            def retain_writer(*args, **kwargs):
                engine = construct(*args, **kwargs)
                if "--index" in engine.spawn[1] and engine.spawn[1][engine.spawn[1].index("--index") + 1] == "1":
                    engine.wlock.acquire()  # Actual close worker cannot acquire its real writer lock.
                    held.append(engine)
                return engine
            try:
                with mock.patch.object(server, "_construct_startup_engine", side_effect=retain_writer), \
                     mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .05), \
                     mock.patch.object(server, "ENGINE_TERM_WAIT_S", .1), \
                     mock.patch.object(server, "ENGINE_KILL_WAIT_S", .1):
                    with self.assertRaises(server.EngineStuck) as caught:
                        server.start_replicas(cfg, [[0], [1]], str(script), [], False)
                failed = caught.exception
                self.assertEqual(len(children), 3)  # first failure, live sibling, retired retry
                self.assertEqual(len(held), 1)
                self.assertIs(failed.startup_engines[1], held[0])
                self.assertEqual(failed.startup_arena, str(arena))
                self.assertIsNotNone(held[0].__dict__.get("_close_owner"))
                self.assertTrue(arena.exists())
                removal.assert_not_called()
            finally:
                for engine in held:
                    engine.wlock.release()
                if failed is not None:
                    self.retire_group(failed.startup_engines)

    def test_actual_child_close_uncertainty_preserves_parent_arena(self):
        with self.group("healthy") as (server, cfg, script, arena, children, removal):
            group = server.start_replicas(cfg, [[0], [1]], str(script), [], False)
            first = group.engines[0]
            first.wlock.acquire()
            try:
                with mock.patch.object(server, "ENGINE_QUIT_WAIT_S", .05), \
                     mock.patch.object(server, "ENGINE_TERM_WAIT_S", .1), \
                     mock.patch.object(server, "ENGINE_KILL_WAIT_S", .1):
                    with self.assertRaises(server.EngineStuck) as caught:
                        group.close()
                self.assertTrue(arena.exists())
                self.assertIs(caught.exception.startup_engines, group.engines)
                self.assertEqual(caught.exception.startup_arena, str(arena))
                self.assertIsNotNone(first.__dict__.get("_close_owner"))
                removal.assert_not_called()
            finally:
                first.wlock.release()
                group.close()
            self.assertFalse(arena.exists())


if __name__ == "__main__":
    unittest.main()

"""Ordinary BCO reservations stay with the epoch that supplied them.

These exercise actual Python coalescing, slot selection, and writer methods with
recorded processes. They do not establish native coalesced-prefill correctness.
"""
import threading
import unittest
from unittest import mock

from serve import server
from serve.test_engine_epoch import batch_engine, replace
from serve.test_engine_writer import SeenLock


class CoalescingEpochTests(unittest.TestCase):
    def configured(self, **changes):
        settings = dict(COALESCE_PREFILL=3, COALESCE_MIN_REQS=3,
                        COALESCE_WAIT_MS=0, COALESCE_TOKENS=128)
        settings.update(changes)
        return mock.patch.multiple(server, **settings)

    @staticmethod
    def followers(engine, *prompts):
        requests = [threading.Event() for _ in prompts]
        engine.wait_lens = [[len(prompt), 0.0, True, prompt, request]
                            for prompt, request in zip(prompts, requests)]
        return requests

    def test_two_followers_send_complete_bco_frames_and_keep_their_own_slots(self):
        engine = batch_engine()
        engine.slot_busy[:] = [False, False]
        epoch = engine._capture_epoch()
        first, second = [7] * 32, [9] * 32
        requests = self.followers(engine, first, second)
        try:
            with self.configured(), engine.ctl:
                engine._coalesce(threading.Event(), [3] * 32, epoch=epoch)
            self.assertEqual(engine.proc.stdin.text,
                             "BCO 0 " + ",".join(map(str, first)) + "\n" +
                             "BCO 1 " + ",".join(map(str, second)) + "\n")
            self.assertEqual(epoch.slots.slot_busy, [True, True])
            self.assertEqual(epoch.slots.slot_claims, {0, 1})
            self.assertEqual([request._strata_preslot for request in requests], [0, 1])
            self.assertTrue(all(request._strata_preslot_epoch is epoch for request in requests))
        finally:
            for request in requests:
                engine._release_preslot(request)
        self.assertEqual(epoch.slots.slot_busy, [False, False])
        self.assertEqual(epoch.slots.slot_claims, set())

    def test_token_budget_excludes_the_second_followers_frame_and_reservation(self):
        engine = batch_engine()
        engine.slot_busy[:] = [False, False]
        epoch = engine._capture_epoch()
        first, second = [7] * 32, [9] * 32
        requests = self.followers(engine, first, second)
        try:
            with self.configured(COALESCE_TOKENS=65), engine.ctl:
                engine._coalesce(threading.Event(), [3] * 32, epoch=epoch)
            self.assertEqual(engine.proc.stdin.text, "BCO 0 " + ",".join(map(str, first)) + "\n")
            self.assertEqual(epoch.slots.slot_busy, [True, False])
            self.assertEqual(epoch.slots.slot_claims, {0})
            self.assertIsNone(getattr(requests[1], "_strata_preslot", None))
        finally:
            for request in requests:
                engine._release_preslot(request)

    def test_releasing_a_coalesced_slot_cannot_free_the_replacements_slot(self):
        engine = batch_engine()
        epoch = engine._capture_epoch()
        request, = self.followers(engine, [7] * 32)
        with self.configured(COALESCE_MIN_REQS=2), engine.ctl:
            engine._coalesce(threading.Event(), [3] * 32, epoch=epoch)
        self.assertEqual(request._strata_preslot, 0)
        replace(engine)
        successor = engine._capture_epoch()
        successor.lines.put("BIPAY successor-owned")
        engine._release_preslot(request)
        engine._release_preslot(request)  # terminal cleanup is idempotent
        self.assertEqual(epoch.slots.slot_busy, [False, True])
        self.assertEqual(epoch.slots.slot_claims, set())
        self.assertEqual(successor.slots.slot_busy, [True, False])
        self.assertEqual(successor.slots.slot_claims, {0})
        self.assertEqual(successor.lines.get_nowait(), "BIPAY successor-owned")
        self.assertIsNone(request._strata_preslot_epoch)

    def test_queued_coalesced_write_cannot_retarget_a_replacement(self):
        engine = batch_engine()
        epoch = engine._capture_epoch()
        old_process = engine.proc
        request, = self.followers(engine, [7] * 32)
        writer = SeenLock()
        engine.wlock = writer
        writer.lock.acquire()
        writer_held = True
        errors = []

        def coalesce():
            try:
                with engine.ctl:
                    engine._coalesce(threading.Event(), [3] * 32, epoch=epoch)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=coalesce)
        try:
            with self.configured(COALESCE_MIN_REQS=2):
                worker.start()
                self.assertTrue(writer.waiting.wait(1), "actual writer must reach its queued boundary")
                # A recorded replacement schedule, not a native restart claim.
                replace(engine)
                successor = engine._capture_epoch()
                writer.lock.release()
                writer_held = False
                worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1, errors)
            self.assertIsInstance(errors[0], server.EngineDied)
            self.assertEqual(old_process.stdin.text, "")
            self.assertEqual(successor.proc.stdin.text, "")
            self.assertIs(request._strata_preslot_epoch, epoch)
            engine._release_preslot(request)
            self.assertEqual(successor.slots.slot_busy, [True, False])
            self.assertEqual(successor.slots.slot_claims, {0})
        finally:
            if writer_held:
                writer.lock.release()
            if worker.ident is not None:
                worker.join(2)
            engine._release_preslot(request)


if __name__ == "__main__":
    unittest.main()

"""Captured slot selection must not use a successor's batch dimensions."""
import queue
import unittest
from serve.test_engine_epoch import batch_engine, replace


def configure(engine, busy, groups):
    count = len(busy)
    engine.batch = count
    engine.slot_busy = list(busy)
    engine.slot_order = list(range(count))
    engine.slot_group = list(groups)
    engine.slot_groups = len(set(groups))
    engine.slot_gs = 0
    engine.slot_q = [queue.Queue() for _ in busy]
    engine.slot_held = [[] for _ in busy]
    engine.slot_used = [0 for _ in busy]
    engine.slot_live = [None for _ in busy]


class EpochSlotCountTests(unittest.TestCase):
    def test_growing_successor_does_not_index_old_slots(self):
        e = batch_engine()
        configure(e, [False, False], [0, 1])
        old = e._capture_epoch()
        replace(e)
        configure(e, [True, True, False, False], [0, 0, 1, 1])
        successor = e._capture_epoch()
        before = list(successor.slots.slot_busy)
        with old.slot_cv:
            self.assertEqual(e.pick_slot([7], epoch=old), 0)
        self.assertEqual(successor.slots.slot_busy, before)
        self.assertEqual(old.slots.slot_busy, [False, False])

    def test_shrinking_successor_does_not_hide_old_group_occupancy(self):
        e = batch_engine()
        # Free slots 0 and 1; old busy slot 2 makes group 0 less desirable.
        configure(e, [False, False, True, False], [0, 1, 0, 1])
        old = e._capture_epoch()
        replace(e)
        configure(e, [True, False], [0, 0])
        successor = e._capture_epoch()
        before = list(successor.slots.slot_busy)
        with old.slot_cv:
            self.assertEqual(e.pick_slot([7], epoch=old), 1)
        self.assertEqual(successor.slots.slot_busy, before)
        self.assertEqual(old.slots.slot_busy, [False, False, True, False])

    def test_captured_cached_prefix_still_wins(self):
        e = batch_engine()
        configure(e, [False, False], [0, 0])
        e.slot_held[1] = [7]
        old = e._capture_epoch()
        replace(e)
        e.batch = 4
        with old.slot_cv:
            self.assertEqual(e.pick_slot([7, 8], epoch=old), 1)


if __name__ == "__main__":
    unittest.main()

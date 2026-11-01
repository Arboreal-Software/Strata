import copy
import json
import threading
import unittest

from serve.scheduler_control import (SchedulerControl, SchedulerProtocolError,
                                     CAPABILITIES, POLICY, REPLY_FIELDS)


READY = {'schema': 1, 'engine_epoch': 17, 'policy': POLICY,
         'max_reply_bytes': 2048, 'capabilities': sorted(CAPABILITIES),
         'limits': dict(min_base=-10, max_base=20, max_activity_boost=5,
                        max_activity_ttl_ms=1000, max_threads=4,
                        max_activity_leases=4, max_id_bytes=32, max_operations=16,
                        max_edges=16, max_frame_bytes=256,
                        max_control_lines=8, metadata_bytes=65536)}


def reply(sequence=1, **changes):
    result = {key: 0 for key in REPLY_FIELDS}
    result.update(schema=1, engine_epoch=17, sequence=sequence, status='ok',
                  thread_id='thread-a', actor_id='', next_expiry_ms=None,
                  thread_generation=1, accepted_revision=3, applied_revision=1,
                  applied_cut=12)
    result.update(changes)
    return 'SCHED_REPLY ' + json.dumps(result) + '\n'


class SchedulerControlTests(unittest.TestCase):
    def client(self, writer, ready=None):
        client = SchedulerControl(writer)
        client.observe('SCHED_READY ' + json.dumps(ready or READY) + '\n')
        return client

    def test_immediate_reply_retains_accepted_and_applied_separately(self):
        frames = []
        def write(frame):
            frames.append(frame)
            client.observe(reply())
        client = self.client(write)
        result = client.command('CREATE', 'thread-a', -3)
        self.assertEqual(frames, ['SCHED 1 17 1 CREATE thread-a -3\n'])
        self.assertEqual((result['accepted_revision'], result['applied_revision']), (3, 1))

    def test_two_callers_serialize_and_pump_does_not_wait_on_command_lock(self):
        frames, outcomes, errors = [], [], []
        first = threading.Event()
        def write(frame):
            frames.append(frame)
            if len(frames) == 1:
                first.set()
            else:
                client.observe(reply(2))
        client = self.client(write)
        def call():
            try: outcomes.append(client.command('SNAPSHOT', 'thread-a', 1, timeout=1))
            except Exception as error: errors.append(error)
        one = threading.Thread(target=call); two = threading.Thread(target=call)
        one.start(); self.assertTrue(first.wait(1)); two.start()
        client.observe(reply(1))
        one.join(2); two.join(2)
        self.assertFalse(one.is_alive() or two.is_alive())
        self.assertFalse(errors)
        self.assertEqual(sorted(x['sequence'] for x in outcomes), [1, 2])
        self.assertEqual([f.split()[3] for f in frames], ['1', '2'])

    def test_invalid_arguments_never_enter_native_transport_or_consume_sequence(self):
        frames = []
        client = self.client(frames.append)
        cases = [('CREATE', 'x\nSTOP', 1), ('CREATE', 'a', True),
                 ('CREATE', 'a', 21), ('BIND', 2, 'abcdef123456', 1, 1, 2, 1),
                 ('BIND', 0, 'ABCDEF123456', 1, 1, 2, 1),
                 ('BOOST', 'a', 1, 'u', 1, 1, 6, 100),
                 ('OPEN', 'a', 1, 1, 'decode'), ('CREATE', 'é', 1)]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(SchedulerProtocolError):
                client.command(*args)
        self.assertEqual(frames, [])
        self.assertEqual(client._sequence, 0)

    def test_timeout_is_ambiguous_and_never_automatically_replayed(self):
        frames = []
        client = self.client(frames.append)
        with self.assertRaisesRegex(SchedulerProtocolError, 'uncertain'):
            client.command('CREATE', 'a', 0, timeout=.001)
        with self.assertRaises(SchedulerProtocolError):
            client.command('CREATE', 'a', 0)
        self.assertEqual(len(frames), 1)
        with self.assertRaises(SchedulerProtocolError):
            client.observe(reply())

    def test_stale_epoch_or_sequence_fails_pending_command(self):
        for changes in ({'engine_epoch': 18}, {'sequence': 2}, {'applied_revision': 4},
                        {'sequence': True}, {'status': 'invented'}):
            with self.subTest(changes=changes):
                client = self.client(lambda frame: client.observe(reply(**changes)))
                with self.assertRaises(SchedulerProtocolError):
                    client.command('SNAPSHOT', 'thread-a', 1)
                with self.assertRaises(SchedulerProtocolError):
                    client.wait_ready(1)

    def test_unsolicited_reply_and_duplicate_startup_fail_closed(self):
        for line in (reply(), 'SCHED_READY ' + json.dumps(READY)):
            client = self.client(lambda _: None)
            with self.assertRaises(SchedulerProtocolError): client.observe(line)

    def test_startup_rejects_omissions_unknown_policy_bool_and_excessive_bounds(self):
        cases = []
        for key, value in [('policy', 'other'), ('engine_epoch', True),
                           ('max_reply_bytes', 4096), ('capabilities', [])]:
            data = copy.deepcopy(READY); data[key] = value; cases.append(data)
        data = copy.deepcopy(READY); del data['limits']['max_edges']; cases.append(data)
        data = copy.deepcopy(READY); data['limits']['max_id_bytes'] = 10000; cases.append(data)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(SchedulerProtocolError):
                self.client(lambda _: None, data)

    def test_raw_json_duplicates_and_oversize_are_rejected(self):
        for line in ('SCHED_READY {"schema":1,"schema":1}', 'SCHED_READY ' + 'x' * 2050):
            client = SchedulerControl(lambda _: None)
            with self.assertRaises(SchedulerProtocolError): client.observe(line)

    def test_close_wakes_pending_caller_and_token_lines_remain_unconsumed(self):
        started = threading.Event(); errors = []
        client = self.client(lambda _: started.set())
        self.assertFalse(client.observe('BT 0 123\n'))
        def call():
            try: client.command('SNAPSHOT', 'thread-a', 1, timeout=1)
            except SchedulerProtocolError as error: errors.append(str(error))
        worker = threading.Thread(target=call); worker.start()
        self.assertTrue(started.wait(1)); client.close(); worker.join(1)
        self.assertFalse(worker.is_alive()); self.assertEqual(len(errors), 1)

    def test_native_validation_refusal_is_a_real_reply_not_transport_success(self):
        client = self.client(lambda frame: client.observe(reply(int(frame.split()[3]), status='stale_version')))
        self.assertEqual(client.command('BASE', 'thread-a', 1, 0, 3)['status'], 'stale_version')
        self.assertEqual(client.command('SNAPSHOT', 'thread-a', 1)['sequence'], 2)


if __name__ == '__main__':
    unittest.main()

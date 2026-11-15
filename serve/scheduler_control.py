"""Private, epoch-bound scheduler control transport for the serving manager.

This is not a public command-forwarding API. The manager must authenticate thread
and actor ownership before calling it. A command timeout poisons this channel:
the engine may have accepted the mutation, so silently retrying is unsafe.
"""
import json
import re
import threading
import time


POLICY = 'strict-priority-tier-then-class-lottery-v1'
U64 = (1 << 64) - 1
ID = re.compile(r'[A-Za-z0-9_.:-]+', re.ASCII)
LIMITS = ('min_base', 'max_base', 'max_activity_boost', 'max_activity_ttl_ms',
          'max_threads', 'max_activity_leases', 'max_id_bytes', 'max_operations',
          'max_edges', 'max_frame_bytes', 'max_control_lines', 'metadata_bytes')
CAPABILITIES = {'managed_threads', 'managed_activity', 'dependencies',
                'request_bindings', 'accepted_applied', 'idle_expiry',
                'strict_tier_class_lottery'}
STATUSES = {'ok', 'invalid_frame', 'stale_engine', 'stale_sequence',
            'sequence_exhausted', 'unknown_generation', 'stale_version',
            'invalid_priority', 'invalid_time', 'limit_exceeded', 'closed',
            'invalid_acknowledgement', 'invalid_argument', 'cycle', 'terminal',
            'stale_priority', 'allocation_failed', 'not_owned', 'slot_busy', 'not_ready'}
REPLY_FIELDS = {'schema', 'engine_epoch', 'sequence', 'status', 'graph_revision',
                'thread_id', 'thread_generation', 'base', 'base_version',
                'effective_priority', 'activity_boost', 'accepted_revision',
                'applied_revision', 'applied_cut', 'actor_id', 'actor_incarnation',
                'lease_version', 'operation_serial', 'result_revision', 'next_expiry_ms'}
# Typed internal arguments; never concatenate caller-supplied wire text.
GRAMMAR = {
    'CREATE': ('id', 'base'), 'BASE': ('id', 'u', 'u', 'base'),
    'ACTOR': ('id', 'u', 'id', 'u'),
    'ACTOR_SNAPSHOT': ('id', 'u', 'id', 'u'),
    'BOOST': ('id', 'u', 'id', 'u', 'u', 'boost', 'ttl'),
    'DISCONNECT': ('id', 'u', 'id', 'u', 'u'),
    'RETIRE_ACTOR': ('id', 'u', 'id', 'u', 'u'),
    'OPEN': ('id', 'u', 'u', 'class'), 'EDGE': ('u', 'u', 'u', 'u', 'u', 'bit'),
    'CANCEL': ('u', 'u', 'u'), 'RETIRE': ('u', 'u', 'u'),
    'BIND': ('bit', 'request', 'u', 'u', 'u', 'u'),
    'SNAPSHOT': ('id', 'u'), 'CLOSE': ('id', 'u', 'u'),
}


class SchedulerProtocolError(RuntimeError):
    pass


def _need(ok, message):
    if not ok:
        raise SchedulerProtocolError(message)


def _integer(value, low=0, high=U64):
    return type(value) is int and low <= value <= high


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, 'Duplicate scheduler JSON field')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, TypeError, RecursionError) as error:
        raise SchedulerProtocolError('Malformed scheduler JSON') from error
    _need(type(value) is dict, 'Scheduler record must be an object')
    return value


class SchedulerControl:
    """One actual engine epoch and at most one outstanding control command.

    `write_line` must serialize with the manager's other native stdin writes.
    The engine stdout pump calls observe without waiting for command completion.
    No thread, operation, or activity state is inferred from command acceptance.
    """
    def __init__(self, write_line, on_failure=None):
        self._write = write_line
        self._on_failure = on_failure
        self._condition = threading.Condition()
        self._serial = threading.Lock()
        self._ready = None
        self._failed = None
        self._sequence = 0
        self._pending = None
        self._reply = None

    def close(self, reason='Scheduler engine epoch ended', *, failed=True):
        with self._condition:
            first = self._failed is None
            self._failed = reason
            self._condition.notify_all()
        if first and failed and self._on_failure is not None:
            self._on_failure(reason)

    def _healthy(self):
        _need(self._failed is None, self._failed or 'Scheduler channel failed')

    def observe(self, line):
        """Consume control records only. Invalid/stale records poison this epoch."""
        if not line.startswith(('SCHED_READY ', 'SCHED_REPLY ')):
            return False
        with self._condition:
            try:
                self._healthy()
                _need(len(line.encode('utf-8')) <= 2049, 'Scheduler reply exceeds bound')
                prefix, raw = line.rstrip('\n').split(' ', 1)
                record = _json(raw)
                _need(record.get('schema') == 1 and type(record['schema']) is int,
                      'Unsupported scheduler schema')
                _need(_integer(record.get('engine_epoch'), 1), 'Invalid scheduler engine epoch')
                if prefix == 'SCHED_READY':
                    self._accept_ready(record)
                else:
                    self._accept_reply(record)
                self._condition.notify_all()
                return True
            except (SchedulerProtocolError, KeyError, TypeError, UnicodeError) as error:
                self._failed = str(error)
                self._condition.notify_all()
                raise SchedulerProtocolError(self._failed) from error

    def _accept_ready(self, record):
        _need(self._ready is None and self._pending is None, 'Duplicate scheduler startup')
        _need(set(record) == {'schema', 'engine_epoch', 'policy', 'limits',
                             'max_reply_bytes', 'capabilities'}, 'Invalid scheduler startup fields')
        _need(record['policy'] == POLICY and type(record['max_reply_bytes']) is int
              and record['max_reply_bytes'] == 2048, 'Unsupported scheduler policy/reply bound')
        caps = record['capabilities']
        _need(type(caps) is list and all(type(c) is str for c in caps)
              and len(caps) == len(CAPABILITIES) and set(caps) == CAPABILITIES,
              'Scheduler capabilities differ')
        limits = record['limits']
        _need(type(limits) is dict and set(limits) == set(LIMITS), 'Scheduler limits incomplete')
        _need(all(_integer(limits[k], -(1 << 31), (1 << 31) - 1)
                  for k in ('min_base', 'max_base')) and limits['min_base'] <= limits['max_base'],
              'Invalid scheduler base bounds')
        _need(_integer(limits['max_activity_boost'], 0, (1 << 31) - 1), 'Invalid activity bound')
        _need(all(_integer(limits[k], 1) for k in LIMITS
                  if k not in ('min_base', 'max_base', 'max_activity_boost')), 'Invalid scheduler limits')
        _need(limits['max_id_bytes'] <= 256 and limits['max_frame_bytes'] <= 65536,
              'Scheduler wire limits exceed manager bounds')
        self._ready = record

    def _accept_reply(self, record):
        _need(self._ready is not None and self._pending is not None and self._reply is None,
              'Unsolicited or duplicate scheduler reply')
        _need(set(record) == REPLY_FIELDS, 'Scheduler reply fields differ')
        _need(record['engine_epoch'] == self._ready['engine_epoch']
              and _integer(record['sequence'], 1) and record['sequence'] == self._pending,
              'Stale scheduler epoch/sequence')
        _need(type(record['status']) is str and record['status'] in STATUSES, 'Unknown scheduler status')
        for key in REPLY_FIELDS - {'schema', 'status', 'thread_id', 'actor_id', 'base',
                                   'activity_boost', 'effective_priority', 'next_expiry_ms'}:
            _need(_integer(record[key]), 'Invalid scheduler counter: ' + key)
        for key in ('base', 'activity_boost'):
            _need(_integer(record[key], -(1 << 31), (1 << 31) - 1), 'Invalid scheduler priority')
        _need(_integer(record['effective_priority'], -(1 << 63), (1 << 63) - 1), 'Invalid effective priority')
        for key in ('thread_id', 'actor_id'):
            value = record[key]
            _need(type(value) is str and (value == '' or ID.fullmatch(value))
                  and len(value) <= self._ready['limits']['max_id_bytes'], 'Invalid scheduler identity')
        _need(record['next_expiry_ms'] is None or _integer(record['next_expiry_ms']), 'Invalid expiry hint')
        _need(record['applied_revision'] <= record['accepted_revision'], 'Applied revision exceeds acceptance')
        self._reply = record

    def wait_ready(self, timeout):
        _need(type(timeout) in (int, float) and 0 < timeout <= 60, 'Invalid scheduler startup deadline')
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._ready is None:
                self._healthy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.close('Scheduler startup timed out')
                    self._healthy()
                self._condition.wait(remaining)
            self._healthy()
            return json.loads(json.dumps(self._ready))

    def _token(self, kind, value):
        limits = self._ready['limits']
        if kind == 'id':
            _need(type(value) is str and ID.fullmatch(value) and len(value) <= limits['max_id_bytes'], 'Invalid managed ID')
        elif kind == 'request':
            _need(type(value) is str and re.fullmatch('[0-9a-f]{12}', value), 'Invalid request binding')
        elif kind == 'class':
            _need(value in ('foreground', 'live_repair', 'saved_maintenance'), 'Invalid work class')
        else:
            lo, hi = {'base': (limits['min_base'], limits['max_base']),
                      'boost': (0, limits['max_activity_boost']), 'ttl': (1, limits['max_activity_ttl_ms']),
                      'bit': (0, 1), 'u': (0, U64)}[kind]
            _need(_integer(value, lo, hi), 'Invalid scheduler argument: ' + kind)
        return str(value)

    def command(self, verb, *arguments, timeout=5.0):
        """Internal typed command; caller owns authentication and handle mapping."""
        _need(type(timeout) in (int, float) and 0 < timeout <= 60, 'Invalid scheduler deadline')
        deadline = time.monotonic() + timeout
        if not self._serial.acquire(timeout=max(0.0, deadline - time.monotonic())):
            self.close('Scheduler command serialization timed out; reconcile engine epoch')
            raise SchedulerProtocolError('Scheduler command serialization timed out')
        try:
            with self._condition:
                self._healthy()
                _need(self._ready is not None, 'Scheduler startup not received')
                _need(type(verb) is str and verb in GRAMMAR
                      and len(arguments) == len(GRAMMAR[verb]), 'Invalid scheduler command')
                args = [self._token(k, v) for k, v in zip(GRAMMAR[verb], arguments)]
                _need(self._sequence < U64, 'Scheduler sequence exhausted')
                seq = self._sequence + 1
                frame = ' '.join(['SCHED', '1', str(self._ready['engine_epoch']), str(seq), verb, *args])
                _need(len(frame) <= self._ready['limits']['max_frame_bytes'], 'Scheduler frame exceeds bound')
                self._sequence = seq
                self._pending = seq
                self._reply = None
            try:
                if time.monotonic() >= deadline:
                    raise SchedulerProtocolError('Scheduler deadline exhausted before write')
                self._write(frame + '\n')
                if time.monotonic() >= deadline:
                    raise SchedulerProtocolError('Scheduler mutation outcome uncertain after write deadline')
                with self._condition:
                    while self._reply is None:
                        self._healthy()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise SchedulerProtocolError('Scheduler mutation outcome uncertain after timeout')
                        self._condition.wait(remaining)
                    self._healthy()
                    result = self._reply
                    self._pending = self._reply = None
                    return result
            except BaseException:
                self.close('Scheduler mutation outcome uncertain; reconcile engine epoch')
                raise
        finally:
            self._serial.release()

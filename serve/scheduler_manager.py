"""Owner-only scheduler authority; durable preferences are not native acknowledgements.

Only thread identity/base survive restart. Activity and operation handles are
engine-epoch scoped. No uncertain wire command is replayed.
"""
import json
import collections
import os
from pathlib import Path
import secrets
import threading

from .scheduler_control import SchedulerProtocolError

MAX_THREADS = 128
MAX_STATE_BYTES = 65536
MAX_HANDLES = 1024
U64 = (1 << 64) - 1


class SchedulerManagerError(ValueError):
    pass


class SchedulerRequestCancelled(SchedulerManagerError):
    pass


def need(ok, message):
    if not ok:
        raise SchedulerManagerError(message)


def integer(value, low=0, high=U64):
    return type(value) is int and low <= value <= high


class SchedulerManager:
    def __init__(self, state_path):
        self.path = Path(state_path)
        need(self.path.is_absolute(), 'Scheduler state path must be absolute')
        self.lock = threading.RLock()
        self.revision = 0
        self.threads = {}
        self.default_thread = None
        self.client = None
        self.native = {}
        self.actors = {}
        self.operations = {}
        self.bindings = {}
        self.pending_bindings = {}
        self.result_revision = 0
        self._cancel_condition = threading.Condition()
        self._cancel_pending = set()
        self._cancel_owned = set()
        self._cancel_queue = collections.deque()
        self._cancel_worker = None
        if self.path.exists():
            need(self.path.stat().st_size <= MAX_STATE_BYTES, 'Scheduler registry exceeds bound')
            def pairs(items):
                out = {}
                for key, value in items:
                    need(key not in out, 'Duplicate scheduler registry field')
                    out[key] = value
                return out
            record = json.loads(self.path.read_text(), object_pairs_hook=pairs)
            need(type(record) is dict, 'Invalid scheduler registry')
            need(set(record) == {'schema', 'revision', 'threads', 'default_thread'} and type(record['schema']) is int and record['schema'] == 1,
                 'Invalid scheduler registry')
            need(integer(record['revision']) and type(record['threads']) is dict and
                 len(record['threads']) <= MAX_THREADS, 'Invalid scheduler registry bounds')
            for key, value in record['threads'].items():
                need(self._handle(key, 't_') and type(value) is dict and set(value) == {'base'} and
                     integer(value['base'], -(1 << 31), (1 << 31)-1), 'Invalid saved thread')
            need(record['default_thread'] is None or record['default_thread'] in record['threads'],
                 'Invalid durable default thread')
            self.default_thread = record['default_thread']
            self.revision, self.threads = record['revision'], record['threads']

    @staticmethod
    def _handle(value, prefix):
        return (type(value) is str and value.startswith(prefix) and len(value) == len(prefix)+32 and
                all(c in '0123456789abcdef' for c in value[len(prefix):]))

    def _persist(self, threads, *, default_thread=None):
        need(self.revision < U64, 'Scheduler manager revision exhausted')
        data = json.dumps({'schema': 1, 'revision': self.revision+1, 'threads': threads,
                           'default_thread': default_thread or self.default_thread},
                          sort_keys=True, separators=(',', ':')).encode()
        need(len(data) <= MAX_STATE_BYTES, 'Scheduler registry exceeds bound')
        # Parent directory is explicitly configured by the service owner, never HTTP.
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name+'.'+secrets.token_hex(8)+'.tmp')
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            if self.client is not None:
                self.client.close('Scheduler durable registry update failed')
            raise
        finally:
            temporary.unlink(missing_ok=True)
        self.threads = threads
        self.default_thread = default_thread or self.default_thread
        self.revision += 1

    def attach(self, client):
        with self.lock:
            if self.client is client:
                return
            ready = client.wait_ready(5.0)
            need(ready['limits'].get('max_id_bytes', 64) >= 34, 'Native ID bound cannot hold manager identities')
            need(len(self.threads) <= min(MAX_THREADS, ready['limits']['max_threads']),
                 'Saved threads exceed native bound')
            # Fresh epoch: old actors, bindings, results and applied evidence are invalid.
            previous = self.client
            if previous is not None:
                previous.close('Scheduler epoch superseded; reconcile old ownership')
            self.client = client
            with self._cancel_condition:
                self._cancel_owned.clear()
                self._cancel_queue.clear()
                self._cancel_pending.clear()
            self.native, self.actors, self.operations, self.bindings, self.pending_bindings = {}, {}, {}, {}, {}
            try:
                for thread, desired in self.threads.items():
                    self.native[thread] = self._command('CREATE', thread, desired['base'])
            except BaseException:
                client.close('Scheduler reconstruction incomplete')
                self.client = None
                raise

    def ensure_default(self, initial_base):
        with self.lock:
            if self.default_thread is None:
                need(len(self.threads) < MAX_THREADS, 'Scheduler thread bound reached')
                thread = 't_'+secrets.token_hex(16)
                reply = self._command('CREATE', thread, initial_base)
                self._persist({**self.threads, thread: {'base': initial_base}}, default_thread=thread)
                self.native[thread] = reply
            return self.default_thread

    def _command(self, verb, *args):
        need(self.client is not None, 'Scheduler engine unavailable')
        reply = self.client.command(verb, *args)
        need(reply['status'] == 'ok', 'Native scheduler refused: '+reply['status'])
        if verb in ('CREATE', 'BASE', 'SNAPSHOT', 'ACTOR', 'ACTOR_SNAPSHOT', 'BOOST',
                    'DISCONNECT', 'RETIRE_ACTOR', 'OPEN', 'CLOSE'):
            expected_generation = reply['thread_generation'] if verb == 'CREATE' else args[1]
            if (reply['thread_id'] != args[0] or not integer(reply['thread_generation'], 1) or
                    reply['thread_generation'] != expected_generation):
                self.client.close('Scheduler reply names another managed thread generation')
                raise SchedulerProtocolError('Scheduler managed thread reply mismatch')
        return reply

    def _thread(self, handle):
        need(type(handle) is str and handle in self.native, 'Unknown thread handle')
        return self.native[handle]

    def _operation(self, handle):
        need(type(handle) is str and handle in self.operations, 'Unknown or stale operation handle')
        return self.operations[handle]

    def _open(self, thread, kind):
        snapshot = self._thread(thread)
        need(len(self.operations) < MAX_HANDLES and self.result_revision < U64,
             'Scheduler operation handle bound reached')
        self.result_revision += 1
        reply = self._command('OPEN', thread, snapshot['thread_generation'], self.result_revision, kind)
        handle = 'o_'+secrets.token_hex(16)
        self.operations[handle] = {'thread': thread, 'reply': reply, 'bound': False, 'class': kind}
        return handle

    def _drain_retirements(self):
        import queue
        pending = getattr(self.client, 'retirements', None)
        if pending is not None:
            while True:
                try:
                    self.retired(pending.get_nowait())
                except queue.Empty:
                    break

    def action(self, body):
        with self.lock:
            self._drain_retirements()
            need(type(body) is dict and type(body.get('action')) is str, 'Typed scheduler action required')
            action = body['action']
            fields = {
                'list': set(), 'create': {'base', 'expected_revision'},
                'base': {'thread', 'base', 'expected_revision'},
                'close': {'thread', 'expected_revision'},
                'snapshot': {'thread'}, 'actor': {'thread'},
                'boost': {'actor', 'boost', 'ttl_ms'}, 'disconnect': {'actor'}, 'retire_actor': {'actor'},
                'open': {'thread', 'class'}, 'cancel': {'operation'}, 'retire': {'operation'},
                'edge': {'consumer', 'prerequisite', 'add'},
            }
            need(action in fields and set(body) == fields[action] | {'action'},
                 'Unknown action or unexpected scheduler field')
            if 'expected_revision' in body:
                need(integer(body['expected_revision']) and body['expected_revision'] == self.revision,
                     'Stale scheduler manager revision')
            result = None
            if action == 'list':
                return {'manager_revision': self.revision, 'default_thread': self.default_thread,
                        'desired_threads': [{'thread': thread, 'base': desired['base']}
                                            for thread, desired in self.threads.items()]}
            if action == 'create':
                need(len(self.threads) < MAX_THREADS, 'Scheduler thread bound reached')
                thread = 't_'+secrets.token_hex(16)
                result = self._command('CREATE', thread, body['base'])
                self._persist({**self.threads, thread: {'base': body['base']}})
                self.native[thread] = result
            elif action in ('base', 'close', 'snapshot', 'actor', 'open'):
                thread = body['thread']
                current = self._thread(thread)
                generation = current['thread_generation']
                if action == 'base':
                    result = self._command('BASE', thread, generation, current['base_version'], body['base'])
                    self._persist({**self.threads, thread: {'base': body['base']}})
                    self.native[thread] = result
                elif action == 'close':
                    need(thread != self.default_thread, 'Configured default thread cannot be closed')
                    result = self._command('CLOSE', thread, generation, current['accepted_revision'])
                    desired = dict(self.threads)
                    del desired[thread]
                    self._persist(desired)
                    del self.native[thread]
                    self.actors = {handle: actor for handle, actor in self.actors.items()
                                   if actor['thread'] != thread}
                    self.operations = {handle: op for handle, op in self.operations.items()
                                       if op['thread'] != thread or op['bound']}
                    removed_pending = [request for request, pair in self.pending_bindings.items()
                                       if not all(handle in self.operations for handle in pair)]
                    for request in removed_pending:
                        self.pending_bindings.pop(request)
                    with self._cancel_condition:
                        for request in removed_pending:
                            self._cancel_owned.discard((self.client, request))
                elif action == 'snapshot':
                    result = self._command('SNAPSHOT', thread, generation)
                    self.native[thread] = result
                elif action == 'actor':
                    need(len(self.actors) < MAX_HANDLES, 'Scheduler actor bound reached')
                    handle = 'a_'+secrets.token_hex(16)
                    result = self._command('ACTOR', thread, generation, handle, current['accepted_revision'])
                    self.actors[handle] = {'thread': thread, 'reply': result}
                    self.native[thread] = result
                else:
                    handle = self._open(thread, body['class'])
                    return {'operation': handle, 'manager_revision': self.revision,
                            'native': self.operations[handle]['reply']}
            elif action in ('boost', 'disconnect', 'retire_actor'):
                handle = body['actor']
                need(type(handle) is str and handle in self.actors, 'Unknown or stale actor handle')
                actor = self.actors[handle]
                thread = actor['thread']
                current, previous = self._thread(thread), actor['reply']
                previous = self._command('ACTOR_SNAPSHOT', thread, current['thread_generation'],
                                         handle, previous['actor_incarnation'])
                args = (thread, current['thread_generation'], handle, previous['actor_incarnation'],
                        previous['lease_version'])
                if action == 'boost':
                    result = self._command('BOOST', *args, body['boost'], body['ttl_ms'])
                else:
                    result = self._command(action.upper(), *args)
                self.native[thread] = result
                actor['reply'] = result
                if action == 'retire_actor':
                    del self.actors[handle]
            else:
                # Query graph revision from an authentic thread snapshot before every CAS.
                key = body['consumer'] if action == 'edge' else body['operation']
                operation = self._operation(key)
                thread = operation['thread']
                current = self._thread(thread)
                snapshot = self._command('SNAPSHOT', thread, current['thread_generation'])
                self.native[thread] = snapshot
                record = operation['reply']
                if action == 'edge':
                    prerequisite = self._operation(body['prerequisite'])['reply']
                    need(type(body['add']) is bool, 'Dependency add must be boolean')
                    result = self._command('EDGE', snapshot['graph_revision'], record['operation_serial'],
                                           record['result_revision'], prerequisite['operation_serial'],
                                           prerequisite['result_revision'], int(body['add']))
                else:
                    result = self._command(action.upper(), snapshot['graph_revision'],
                                           record['operation_serial'], record['result_revision'])
                    if action == 'retire':
                        del self.operations[key]
            response = {'manager_revision': self.revision, 'native': result}
            if action == 'create':
                response['thread'] = thread
            if action == 'actor':
                response['actor'] = handle
            return response

    def validate_selection(self, thread, operations=None, *, request=None):
        with self.lock:
            self._thread(thread)
            if operations is not None:
                need(type(operations) in (tuple, list) and len(operations) == 2,
                     'Two issued operation handles required')
                for handle, kind in zip(operations, ('foreground', 'live_repair')):
                    operation = self._operation(handle)
                    need(operation['thread'] == thread and operation['class'] == kind and
                         not operation['bound'], 'Operation is foreign, bound, or wrong class')
                    need(operation.get('reserved_for') in (None, request),
                         'Operation pair is reserved by another request')

    def bind(self, thread, slot, request, *, operations=None):
        with self.lock:
            self._drain_retirements()
            need(request not in self.bindings, 'Request already bound')
            self.validate_selection(thread, operations, request=request)
            try:
                pair = self.pending_bindings.get(request)
                if pair is None:
                    # Publish cleanup ownership before any partial OPEN can refuse.
                    pair = []
                    self.pending_bindings[request] = pair
                    with self._cancel_condition:
                        self._cancel_owned.add((self.client, request))
                    if operations is None:
                        pair.append(self._open(thread, 'foreground'))
                        pair.append(self._open(thread, 'live_repair'))
                    else:
                        pair.extend(operations)
                    pair = tuple(pair)
                    for handle in pair:
                        self.operations[handle]['reserved_for'] = request
                    self.pending_bindings[request] = pair
                foreground, repair = pair
                fg, rp = self.operations[foreground]['reply'], self.operations[repair]['reply']
                result = self.client.command('BIND', slot, request, fg['operation_serial'], fg['result_revision'],
                                             rp['operation_serial'], rp['result_revision'])
                if result['status'] == 'not_ready':
                    return False  # Authentic atomic refusal: no native binding/slot authority.
                if result['status'] == 'terminal':
                    raise SchedulerRequestCancelled('Native graph cancelled this unsubmitted request')
                need(result['status'] == 'ok', 'Native scheduler binding refused: '+result['status'])
                current = self._thread(thread)
                if result['thread_id'] != thread or result['thread_generation'] != current['thread_generation']:
                    raise SchedulerProtocolError('Scheduler binding reply owner mismatch')
            except SchedulerRequestCancelled:
                raise  # Atomic terminal refusal publishes no binding; caller reconciles unbound pair.
            except BaseException:
                self.client.close('Scheduler request binding incomplete')
                raise
            self.operations[foreground]['bound'] = self.operations[repair]['bound'] = True
            self.bindings[request] = self.pending_bindings.pop(request)
            return True

    def cancel_request(self, request, *, client=None):
        with self.lock:
            need(client is None or self.client is client, 'Stale scheduler cancellation epoch')
            binding = self.bindings.get(request)
            if binding is None:
                pending = self.pending_bindings.get(request)
                if pending is not None:
                    for handle in tuple(pending):
                        self._retire_unbound(handle)
                    self.pending_bindings.pop(request, None)
                with self._cancel_condition:
                    self._cancel_owned.discard((self.client, request))
                return
            # One cancellation cascades native physical retirement. Keep handles until
            # actual retirement; no accepted CANCEL is an applied/terminal claim.
            operation = self.operations[binding[1]]
            if operation.get('cancel_sent'):
                return
            record = operation['reply']
            thread = operation['thread']
            if thread not in self.native:
                return  # Accepted CLOSE already cascaded; real BOWN_DONE still owns release.
            current = self._thread(thread)
            snapshot = self._command('SNAPSHOT', thread, current['thread_generation'])
            self.native[thread] = snapshot
            for attempt in range(2):
                result = self.client.command('CANCEL', snapshot['graph_revision'],
                                             record['operation_serial'], record['result_revision'])
                if result['status'] != 'stale_version' or attempt == 1:
                    break
                snapshot = self._command('SNAPSHOT', thread, current['thread_generation'])
                self.native[thread] = snapshot
            need(result['status'] in ('ok', 'terminal', 'unknown_generation', 'stale_version'),
                 'Native scheduler cancellation refused: '+result['status'])
            operation['cancel_reply'] = result  # A refusal is not accepted/applied evidence.
            operation['cancel_sent'] = True

    def _retire_unbound(self, handle):
        operation = self.operations.get(handle)
        if operation is None:
            return
        need(not operation['bound'], 'Cannot retire an owned bound operation')
        thread, record = operation['thread'], operation['reply']
        current = self._thread(thread)
        for verb in ('CANCEL', 'RETIRE'):
            for attempt in range(2):
                snapshot = self._command('SNAPSHOT', thread, current['thread_generation'])
                self.native[thread] = snapshot
                reply = self.client.command(verb, snapshot['graph_revision'],
                                            record['operation_serial'], record['result_revision'])
                if reply['status'] != 'stale_version':
                    break
            allowed = ('ok', 'unknown_generation', 'terminal') if verb == 'CANCEL' else ('ok', 'unknown_generation')
            if reply['status'] not in allowed:
                raise SchedulerProtocolError('Unbound cleanup could not be proven: '+reply['status'])
        self.operations.pop(handle, None)

    def schedule_cancel(self, request, *, client):
        # One bounded queue/worker covers every owned pending pair. Streams never
        # wait for a control reply, and a burst cannot strand a fifth waiter.
        key = (client, request)
        with self._cancel_condition:
            if key not in self._cancel_owned or key in self._cancel_pending:
                return
            need(len(self._cancel_pending) < MAX_HANDLES + 2,
                 'Cancellation inventory exceeds operation ownership bound')
            self._cancel_pending.add(key)
            self._cancel_queue.append(key)
            if self._cancel_worker is None:
                self._cancel_worker = threading.Thread(target=self._reconcile_cancellations,
                                                       name='strata-scheduler-cancel', daemon=True)
                self._cancel_worker.start()
            self._cancel_condition.notify()

    def _reconcile_cancellations(self):
        while True:
            with self._cancel_condition:
                if not self._cancel_queue:
                    self._cancel_worker = None
                    return
                client, request = self._cancel_queue.popleft()
                owned = (client, request) in self._cancel_owned
            try:
                if owned:
                    self.cancel_request(request, client=client)
            except BaseException as error:
                client.close('Scheduler cancellation reconciliation failed: '+str(error))
            finally:
                with self._cancel_condition:
                    self._cancel_pending.discard((client, request))

    def retired(self, request):
        with self.lock:
            for handle in self.bindings.pop(request, ()):
                self.operations.pop(handle, None)
            with self._cancel_condition:
                self._cancel_owned.discard((self.client, request))

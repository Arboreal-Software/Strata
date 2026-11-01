"""Manager authority/durability checks with a recording native boundary."""
import dataclasses
import io
import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from serve.scheduler_manager import SchedulerManager, SchedulerManagerError
from serve.scheduler_control import SchedulerControl, SchedulerProtocolError
from serve import server
from serve.test_engine_epoch import batch_engine, replace
from serve.test_scheduler_control import READY, reply
from types import SimpleNamespace
import copy
def ready():
    value=copy.deepcopy(READY);value["limits"]["max_id_bytes"]=64;value["limits"]["max_frame_bytes"]=4096
    return value


class Native:
    def __init__(self, epoch=1):
        self.epoch=epoch; self.calls=[]; self.seq=0; self.graph=1; self.closed=None
        self.retirements=queue.SimpleQueue(); self.threads={}; self.inc=0; self.operations={}
    def wait_ready(self, timeout):
        return {'engine_epoch':self.epoch,'limits':{'max_threads':128}}
    def close(self, reason): self.closed=reason
    def command(self, verb, *args, timeout=5.0):
        if self.closed: raise SchedulerProtocolError(self.closed)
        self.calls.append((verb,*args)); self.seq+=1
        if verb=='CREATE':
            self.inc+=1
            self.threads[args[0]]={'thread_generation':self.epoch*100+self.inc,'base':args[1],
                'base_version':1,'accepted_revision':1,'applied_revision':0,'applied_cut':0}
        thread=args[0] if verb in ('CREATE','BASE','SNAPSHOT','ACTOR','BOOST','DISCONNECT','RETIRE_ACTOR','OPEN','CLOSE') else next(iter(self.threads))
        if verb=='OPEN':self.operations[self.seq]=thread
        if verb=='BIND':thread=self.operations[args[2]]
        row=self.threads[thread]
        if verb=='BASE': row.update(base=args[3],base_version=row['base_version']+1,accepted_revision=row['accepted_revision']+1)
        if verb=='ACTOR': row['accepted_revision']+=1
        self.graph+=1
        return {**row,'engine_epoch':self.epoch,'thread_id':thread,'graph_revision':self.graph,
            'status':'ok','operation_serial':args[1] if verb in ('CANCEL','EDGE','RETIRE') else self.seq,'result_revision':args[2] if verb in ('OPEN','CANCEL','EDGE','RETIRE') else 0,
            'actor_incarnation':self.seq,'lease_version':self.seq}


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'registry.json'; self.m=SchedulerManager(self.path)
        self.n=Native(); self.m.attach(self.n)
        self.t=self.m.ensure_default(3)
    def test_restart_recreates_desired_base_and_invalidates_activity_and_operations(self):
        self.m.action({'action':'base','thread':self.t,'base':7,'expected_revision':1})
        actor=self.m.action({'action':'actor','thread':self.t})['actor']
        op=self.m.action({'action':'open','thread':self.t,'class':'foreground'})['operation']
        self.m.bind(self.t,0,'aaaaaaaaaaaa')
        fresh=SchedulerManager(self.path); native=Native(2); fresh.attach(native)
        self.assertEqual(native.calls,[('CREATE',self.t,7)])
        self.assertEqual(fresh.ensure_default(3),self.t)
        self.assertEqual(fresh.native[self.t]['base'],7)
        for body in ({'action':'disconnect','actor':actor},{'action':'cancel','operation':op}):
            with self.assertRaises(SchedulerManagerError):fresh.action(body)
        self.assertFalse(fresh.bindings)
    def test_revision_cas_and_raw_native_authority_refused(self):
        before=list(self.n.calls)
        for body in ({'action':'base','thread':self.t,'base':8,'expected_revision':0},
                     {'action':'SCHED','frame':'BASE x'},
                     {'action':'snapshot','thread':self.t,'thread_generation':101}):
            with self.assertRaises(SchedulerManagerError):self.m.action(body)
        self.assertEqual(self.n.calls,before)
    def test_concurrent_base_cas_has_one_winner(self):
        results=[]
        def change(base):
            try:self.m.action({'action':'base','thread':self.t,'base':base,'expected_revision':1});results.append(True)
            except SchedulerManagerError:results.append(False)
        threads=[threading.Thread(target=change,args=(value,))for value in (4,5)]
        for thread in threads:thread.start()
        for thread in threads:thread.join()
        self.assertEqual(sorted(results),[False,True]);self.assertEqual(json.loads(self.path.read_text())['revision'],2)
    def test_two_requests_same_thread_get_distinct_authentic_binding_ops(self):
        self.m.bind(self.t,0,'aaaaaaaaaaaa');self.m.bind(self.t,1,'bbbbbbbbbbbb')
        binds=[call for call in self.n.calls if call[0]=='BIND']
        self.assertEqual(len(binds),2);self.assertNotEqual(binds[0][3:],binds[1][3:])
        self.m.cancel_request('aaaaaaaaaaaa',client=self.n)
        self.m.cancel_request('aaaaaaaaaaaa',client=self.n)
        self.assertEqual(sum(call[0]=='CANCEL'for call in self.n.calls),1)
        self.n.retirements.put('aaaaaaaaaaaa');self.m.action({'action':'snapshot','thread':self.t})
        self.assertNotIn('aaaaaaaaaaaa',self.m.bindings);self.assertIn('bbbbbbbbbbbb',self.m.bindings)
    def test_old_cancellation_cannot_touch_replacement_epoch(self):
        old=self.n;self.m.bind(self.t,0,'aaaaaaaaaaaa');new=Native(2);self.m.attach(new)
        with self.assertRaises(SchedulerManagerError):self.m.cancel_request('aaaaaaaaaaaa',client=old)
        self.assertFalse(any(c[0]=='CANCEL'for c in new.calls))
    def test_failed_durable_commit_poisoned_and_not_acknowledged(self):
        with mock.patch('serve.scheduler_manager.os.replace',side_effect=OSError('disk')):
            with self.assertRaises(OSError):self.m.action({'action':'base','thread':self.t,'base':9,'expected_revision':1})
        self.assertIsNotNone(self.n.closed)
        self.assertEqual(SchedulerManager(self.path).threads[self.t]['base'],3)
    def test_owner_auth_and_default_thread_required_config(self):
        svc=object.__new__(server.Service);svc.api_key=''
        with self.assertRaisesRegex(ValueError,'authenticated'):svc.scheduler_manager()
    def test_actual_typed_client_manager_roundtrip_and_restart(self):
        from serve.scheduler_control import REPLY_FIELDS
        native=Native(17)
        def write(frame):
            fields=frame.split();verb=fields[4];args=fields[5:]
            kinds=__import__('serve.scheduler_control',fromlist=['GRAMMAR']).GRAMMAR[verb]
            args=[int(value)if kind in ('u','base','bit','boost','ttl')else value
                  for kind,value in zip(kinds,args)]
            actual=native.command(verb,*args)
            record={field:0 for field in REPLY_FIELDS}
            record.update(actual,schema=1,engine_epoch=17,sequence=int(fields[3]),actor_id='',next_expiry_ms=None)
            client.observe('SCHED_REPLY '+json.dumps(record)+'\n')
        client=SchedulerControl(write);client.retirements=queue.SimpleQueue()
        client.observe('SCHED_READY '+json.dumps(ready())+'\n')
        self.m.attach(client)
        result=self.m.action({'action':'snapshot','thread':self.t})
        self.assertEqual(result['native']['engine_epoch'],17)
        self.m.bind(self.t,0,'aaaaaaaaaaaa')
        self.assertEqual(sum(row[0]=='BIND'for row in native.calls),1)

    def test_explicit_operation_pair_preserves_dependency_graph_binding(self):
        fg=self.m.action({'action':'open','thread':self.t,'class':'foreground'})['operation']
        rp=self.m.action({'action':'open','thread':self.t,'class':'live_repair'})['operation']
        wait=self.m.action({'action':'open','thread':self.t,'class':'foreground'})['operation']
        self.m.action({'action':'edge','consumer':fg,'prerequisite':wait,'add':True})
        self.m.bind(self.t,0,'aaaaaaaaaaaa',operations=(fg,rp))
        self.assertEqual(self.n.calls[-1][0],'BIND')
        self.assertTrue(self.m.operations[fg]['bound'])

    def test_actual_route_rejects_unauthorized_before_body_or_manager(self):
        svc=SimpleNamespace(api_key='secret',scheduler_manager=mock.Mock())
        handler=object.__new__(server.make_handler(svc))
        handler.headers={};handler.path='/v1/scheduler';handler._json=mock.Mock()
        handler._body=mock.Mock(side_effect=AssertionError('unauthorized body read'))
        handler._post();self.assertEqual(handler._json.call_args.args[0],401)
        svc.scheduler_manager.assert_not_called()

    def test_ordinary_request_default_reaches_actual_bind_before_bgen(self):
        from serve.two_slot_receipts import OwnedSlotObservations
        e=batch_engine();epoch=e._capture_epoch()
        epoch=dataclasses.replace(epoch,scheduler=self.n,strict_owned=True)
        epoch.slots.owned_observations=OwnedSlotObservations();e._epoch=epoch
        svc=object.__new__(server.Service);svc.api_key='owner';svc.engine=e
        svc.status_lock=threading.RLock();svc._scheduler_manager=self.m;e.scheduler_manager=self.m
        cancel=threading.Event();cancel._strata_request_id='aaaaaaaaaaaa'
        with mock.patch.dict('os.environ',{'STRATA_SERVING_THREAD_STATE_V1':str(self.path),
                                        'STRATA_SERVING_THREAD_DEFAULT_BASE_V1':'3'}):
            svc.scheduler_request({},cancel)
        self.assertEqual(cancel._strata_scheduler_thread,self.t)
        original=e._send
        def admit(frame,**kw):
            if frame.startswith('BGEN '):
                self.assertEqual(self.n.calls[-1][0],'BIND')
                e.lines.put('BADM 0 1');e.slot_q[0].put('BT 0 7');e.slot_q[0].put('BDONE 0 1 length 0')
            original(frame,**kw)
        e._send=admit
        self.assertEqual(list(e.generate_batched([1,2],1,{},cancel)),[7])
        self.assertIn('BGEN 0 1 http_request=aaaaaaaaaaaa 1,2',e.proc.stdin.text)

    def test_blocked_consumer_releases_admission_and_slot_for_prerequisite_worker(self):
        from serve.two_slot_receipts import OwnedSlotObservations
        e=batch_engine();e.slot_busy=[False,False];epoch=e._capture_epoch()
        epoch=dataclasses.replace(epoch,scheduler=self.n,strict_owned=True)
        epoch.slots.owned_observations=OwnedSlotObservations();e._epoch=epoch;e.scheduler_manager=self.m
        fg=self.m.action({'action':'open','thread':self.t,'class':'foreground'})['operation']
        rp=self.m.action({'action':'open','thread':self.t,'class':'live_repair'})['operation']
        pf=self.m.action({'action':'open','thread':self.t,'class':'foreground'})['operation']
        pr=self.m.action({'action':'open','thread':self.t,'class':'live_repair'})['operation']
        self.m.action({'action':'edge','consumer':fg,'prerequisite':pf,'add':True})
        prerequisite_done=threading.Event();consumer_waiting=threading.Event();commands=[];errors=[];outputs={}
        command=self.n.command
        def native(verb,*args,timeout=5.0):
            result=command(verb,*args)
            if verb=='BIND' and args[1]=='aaaaaaaaaaaa' and not prerequisite_done.is_set():
                result['status']='not_ready'
            return result
        self.n.command=native
        original=e._send
        def admit(frame,**kw):
            commands.append(frame)
            if frame.startswith('BGEN '):
                slot=int(frame.split()[1])
                e.lines.put('BADM %d 1'%slot)
                e.slot_q[slot].put('BT %d 7'%slot);e.slot_q[slot].put('BDONE %d 1 length 0'%slot)
            original(frame,**kw)
        e._send=admit
        cancels=[]
        def worker(request,pair):
            cancel=threading.Event();cancels.append(cancel)
            cancel._strata_request_id=request;cancel._strata_scheduler_thread=self.t
            cancel._strata_scheduler_manager=self.m;cancel._strata_scheduler_operations=pair
            try:
                outputs[request]=[]
                for item in e.generate_batched([1,2],1,{},cancel):
                    outputs[request].append(item)
                    if request=='aaaaaaaaaaaa' and item is None:consumer_waiting.set()
                if request=='bbbbbbbbbbbb':prerequisite_done.set()
            except BaseException as error:errors.append(error)
        consumer=threading.Thread(target=worker,args=('aaaaaaaaaaaa',(fg,rp)))
        prerequisite=threading.Thread(target=worker,args=('bbbbbbbbbbbb',(pf,pr)))
        consumer.start()
        try:
            self.assertTrue(consumer_waiting.wait(1))
            self.assertFalse(any(frame.startswith('BGEN ')for frame in commands))
            self.assertFalse(any(e.slot_busy));self.assertFalse(e.ctl.locked())
            prerequisite.start();prerequisite.join(1);consumer.join(1)
            self.assertFalse(prerequisite.is_alive());self.assertFalse(consumer.is_alive())
            self.assertEqual(errors,[]);self.assertIn(7,outputs['aaaaaaaaaaaa']);self.assertEqual(outputs['bbbbbbbbbbbb'],[7])
            requests=[frame.split('http_request=')[1][:12]for frame in commands if frame.startswith('BGEN ')]
            self.assertEqual(requests,['bbbbbbbbbbbb','aaaaaaaaaaaa'])
        finally:
            prerequisite_done.set()
            for cancel in cancels:cancel.set()
            consumer.join(1)
            if prerequisite.ident is not None:prerequisite.join(1)

    def test_activity_refresh_observes_actual_expired_lease_version(self):
        handle=self.m.action({'action':'actor','thread':self.t})['actor']
        incarnation=self.m.actors[handle]['reply']['actor_incarnation'];observed=[]
        native=self.n.command
        def changed(verb,*args,timeout=5.0):
            out=native(verb,*args)
            if verb=='ACTOR_SNAPSHOT':
                out.update(actor_incarnation=incarnation,lease_version=91)
            elif verb in ('BOOST','DISCONNECT'):
                observed.append(args[4]);out.update(actor_incarnation=incarnation,lease_version=92)
            return out
        self.n.command=changed
        self.m.action({'action':'boost','actor':handle,'boost':2,'ttl_ms':10})
        self.m.action({'action':'disconnect','actor':handle})
        self.assertEqual(observed,[91,91])
        self.assertEqual(sum(call[0]=='ACTOR_SNAPSHOT'for call in self.n.calls),2)

    def _retry_bind_while_peer_has_control(self, ending):
        """Actual admission generator and lock wait, simulated native boundary.

        not_ready yields with no control. A peer then occupies control during
        prefill while another physical slot is free. The retry must reacquire,
        rather than issuing a synchronous BIND behind that peer's native work.
        Events distinguish the first real lock attempt from an unlocked BIND;
        there is no CUDA, timeout enlargement or sleep-based race assertion.
        """
        from serve.two_slot_receipts import OwnedSlotObservations
        e=batch_engine();del e._take_control  # Exercise the real cancellation/deadline-aware method.
        probe=threading.Event();observation=[]
        class Control:
            def __init__(lock):lock.raw=threading.Lock();lock.owner=None
            def acquire(lock,*args,**kwargs):
                if lock.raw.locked() and lock.owner!=threading.get_ident() and not probe.is_set():
                    observation.append('waiting_for_peer_control');probe.set()
                ok=lock.raw.acquire(*args,**kwargs)
                if ok:lock.owner=threading.get_ident()
                return ok
            def release(lock):
                if lock.owner!=threading.get_ident():raise AssertionError('Released another request control')
                lock.owner=None;lock.raw.release()
            def locked(lock):return lock.raw.locked()
        e.ctl=Control();e.slot_busy=[False,False]
        epoch=dataclasses.replace(e._capture_epoch(),scheduler=self.n,strict_owned=True)
        epoch.slots.owned_observations=OwnedSlotObservations();e._epoch=epoch;e.scheduler_manager=self.m
        pairs=[]
        for _ in range(2):
            pairs.append(tuple(self.m.action({'action':'open','thread':self.t,'class':kind})['operation'] for kind in ('foreground','live_repair')))
        self.m.action({'action':'edge','consumer':pairs[0][0],'prerequisite':pairs[1][0],'add':True})
        ready=threading.Event();binds=[];frames=[];errors=[];outputs=[]
        native=self.n.command
        def command(verb,*args,timeout=5.0):
            if verb=='BIND':
                binds.append(args[1])
                if e.ctl.owner!=threading.get_ident():
                    if not probe.is_set():observation.append('bind_without_control');probe.set()
                    raise AssertionError('BIND without owning admission control')
            result=native(verb,*args)
            if verb=='BIND' and args[1]=='aaaaaaaaaaaa' and not ready.is_set():result['status']='not_ready'
            return result
        self.n.command=command
        original=e._send;peer_slot=[]
        def send(frame,**kwargs):
            frames.append(frame)
            if frame.startswith('BGEN '):
                slot=int(frame.split()[1])
                if 'http_request=bbbbbbbbbbbb' in frame:
                    peer_slot.append(slot);e.lines.put('PP 1 2 0 1.0')
                else:
                    e.lines.put('BADM %d 1'%slot);e.slot_q[slot].put('BT %d 7'%slot);e.slot_q[slot].put('BDONE %d 1 length 0'%slot)
            original(frame,**kwargs)
        e._send=send
        def cancel_for(key,pair):
            cancel=threading.Event();cancel._strata_request_id=key;cancel._strata_scheduler_thread=self.t
            cancel._strata_scheduler_manager=self.m;cancel._strata_scheduler_operations=pair;return cancel
        cancel=cancel_for('aaaaaaaaaaaa',pairs[0]);peer_cancel=cancel_for('bbbbbbbbbbbb',pairs[1])
        consumer=e.generate_batched([1,2],1,{},cancel);peer=e.generate_batched([3,4],1,{},peer_cancel)
        self.assertIsNone(next(consumer));self.assertEqual(binds,['aaaaaaaaaaaa']);self.assertFalse(e.ctl.locked());self.assertFalse(any(e.slot_busy))
        self.assertIsNone(next(peer));self.assertEqual(len(peer_slot),1);self.assertTrue(e.ctl.locked());self.assertEqual(e.ctl.owner,threading.get_ident())
        def resume():
            try:outputs.extend(consumer)
            except BaseException as error:errors.append(error)
        worker=threading.Thread(target=resume);worker.start();peer_finished=False
        try:
            self.assertTrue(probe.wait(1),'Retry reached neither lock acquisition nor BIND')
            self.assertEqual(observation,['waiting_for_peer_control'])
            self.assertEqual(binds,['aaaaaaaaaaaa','bbbbbbbbbbbb'])
            if ending=='cancel':cancel.set()
            elif ending=='deadline':cancel._strata_queue_deadline=server.time.monotonic()-1
            if ending!='continue':
                worker.join(2);self.assertFalse(worker.is_alive(),'Canceled/expired retry did not leave control wait')
                self.assertEqual(e.ctl.owner,threading.get_ident());self.assertTrue(e.slot_busy[peer_slot[0]])
            ready.set();slot=peer_slot[0]
            e.lines.put('BADM %d 1'%slot);e.slot_q[slot].put('BT %d 9'%slot);e.slot_q[slot].put('BDONE %d 1 length 0'%slot)
            self.assertEqual(list(peer),[9]);peer_finished=True
            worker.join(2);self.assertFalse(worker.is_alive())
            if ending=='continue':self.assertEqual(errors,[]);self.assertEqual(outputs,[7]);self.assertEqual(binds,['aaaaaaaaaaaa','bbbbbbbbbbbb','aaaaaaaaaaaa'])
            elif ending=='cancel':self.assertEqual(errors,[]);self.assertEqual(outputs,[])
            else:self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],server.QueueWaitExpired);self.assertEqual(outputs,[])
            self.assertFalse(any(frame.startswith(('BCANCEL','BSTOP')) for frame in frames))
            self.assertEqual(sum('http_request=aaaaaaaaaaaa' in frame for frame in frames),1 if ending=='continue' else 0)
            self.assertFalse(e.ctl.locked());self.assertFalse(any(e.slot_busy));self.assertIsNone(self.n.closed)
        finally:
            ready.set();cancel.set()
            if not peer_finished:
                slot=peer_slot[0];e.lines.put('BADM %d 1'%slot);e.slot_q[slot].put('BT %d 9'%slot);e.slot_q[slot].put('BDONE %d 1 length 0'%slot)
                list(peer)
            worker.join(2)
            if not worker.is_alive():consumer.close()
            with self.m._cancel_condition:cleanup=self.m._cancel_worker
            if cleanup is not None:cleanup.join(1)

    def test_not_ready_free_slot_retry_waits_for_peer_control_then_progresses(self):
        self._retry_bind_while_peer_has_control('continue')

    def test_not_ready_retry_cancel_does_not_bind_or_release_peer_control(self):
        self._retry_bind_while_peer_has_control('cancel')

    def test_not_ready_retry_queue_expiry_does_not_bind_or_release_peer_control(self):
        self._retry_bind_while_peer_has_control('deadline')

    def test_close_releases_actor_handles_without_forging_bound_retirement(self):
        thread=self.m.action({'action':'create','base':2,'expected_revision':1})['thread']
        actor=self.m.action({'action':'actor','thread':thread})['actor']
        self.m.bind(thread,0,'aaaaaaaaaaaa')
        self.m.action({'action':'close','thread':thread,'expected_revision':2})
        self.assertNotIn(actor,self.m.actors);self.assertIn('aaaaaaaaaaaa',self.m.bindings)
        self.m.cancel_request('aaaaaaaaaaaa',client=self.n)
        self.assertIn('aaaaaaaaaaaa',self.m.bindings)
        self.n.retirements.put('aaaaaaaaaaaa');self.m.action({'action':'list'})
        self.assertNotIn('aaaaaaaaaaaa',self.m.bindings)

    def test_foreign_reply_generation_poisoned_not_published(self):
        native=self.n.command
        def foreign(verb,*args,timeout=5.0):
            out=native(verb,*args)
            if verb=='SNAPSHOT':out['thread_generation']+=1
            return out
        self.n.command=foreign
        with self.assertRaises(SchedulerProtocolError):self.m.action({'action':'snapshot','thread':self.t})
        self.assertIsNotNone(self.n.closed)

    def test_duplicate_consumer_worker_cannot_steal_pending_pair_or_poison_peer(self):
        from serve.two_slot_receipts import OwnedSlotObservations
        e=batch_engine();e.slot_busy=[False,True];epoch=e._capture_epoch()
        epoch=dataclasses.replace(epoch,scheduler=self.n,strict_owned=True)
        epoch.slots.owned_observations=OwnedSlotObservations();e._epoch=epoch;e.scheduler_manager=self.m
        fg=self.m.action({'action':'open','thread':self.t,'class':'foreground'})['operation']
        rp=self.m.action({'action':'open','thread':self.t,'class':'live_repair'})['operation']
        paused=threading.Event();release=threading.Event();ready_to_bind=threading.Event();errors=[]
        native=self.n.command
        def blocked(verb,*args,timeout=5.0):
            out=native(verb,*args)
            if verb=='BIND' and not ready_to_bind.is_set():out['status']='not_ready'
            return out
        self.n.command=blocked
        original=e._send
        def admit(frame,**kw):
            if frame.startswith('BGEN '):
                slot=int(frame.split()[1]);e.lines.put('BADM %d 1'%slot)
                e.slot_q[slot].put('BT %d 7'%slot);e.slot_q[slot].put('BDONE %d 1 length 0'%slot)
            original(frame,**kw)
        e._send=admit
        def event(request):
            cancel=threading.Event();cancel._strata_request_id=request
            cancel._strata_scheduler_thread=self.t;cancel._strata_scheduler_manager=self.m
            cancel._strata_scheduler_operations=(fg,rp);return cancel
        first_cancel=event('aaaaaaaaaaaa');duplicate_cancel=event('cccccccccccc')
        def first():
            try:
                for value in e.generate_batched([1,2],1,{},first_cancel):
                    if value is None:paused.set();release.wait(1)
            except BaseException as error:errors.append(error)
        worker=threading.Thread(target=first);worker.start()
        try:
            self.assertTrue(paused.wait(1));before=list(self.n.calls)
            with self.assertRaisesRegex(SchedulerManagerError,'reserved'):
                list(e.generate_batched([1,2],1,{},duplicate_cancel))
            self.assertEqual(self.n.calls,before);self.assertIsNone(self.n.closed)
            self.assertTrue(e.slot_busy[1]);self.assertEqual(self.m.pending_bindings['aaaaaaaaaaaa'],(fg,rp))
            self.assertEqual(self.m.operations[fg]['reserved_for'],'aaaaaaaaaaaa')
            ready_to_bind.set();release.set();worker.join(1)
            self.assertFalse(worker.is_alive());self.assertEqual(errors,[])
            self.assertIn('aaaaaaaaaaaa',self.m.bindings)
        finally:
            ready_to_bind.set();first_cancel.set();release.set();worker.join(1)

    def test_owner_cancelled_waiter_is_local_and_retires_unbound_pair(self):
        from serve.two_slot_receipts import OwnedSlotObservations
        e=batch_engine();epoch=e._capture_epoch()
        epoch=dataclasses.replace(epoch,scheduler=self.n,strict_owned=True)
        epoch.slots.owned_observations=OwnedSlotObservations();e._epoch=epoch;e.scheduler_manager=self.m
        self.m.bind(self.t,1,'bbbbbbbbbbbb')
        fg=self.m.action({'action':'open','thread':self.t,'class':'foreground'})['operation']
        rp=self.m.action({'action':'open','thread':self.t,'class':'live_repair'})['operation']
        serial=self.m.operations[fg]['reply']['operation_serial'];terminal=[False]
        native=self.n.command
        def blocked(verb,*args,timeout=5.0):
            out=native(verb,*args)
            if verb=='CANCEL' and args[1]==serial:
                if terminal[0]:out['status']='terminal'
                terminal[0]=True
            if verb=='BIND' and args[1]=='aaaaaaaaaaaa':out['status']='terminal'if terminal[0]else'not_ready'
            return out
        self.n.command=blocked
        cancel=threading.Event();cancel._strata_request_id='aaaaaaaaaaaa'
        cancel._strata_scheduler_manager=self.m;cancel._strata_scheduler_thread=self.t
        cancel._strata_scheduler_operations=(fg,rp)
        generator=e.generate_batched([1,2],1,{},cancel)
        self.assertIsNone(next(generator));self.assertFalse(e.ctl.locked())
        self.m.action({'action':'cancel','operation':fg})
        self.assertEqual(list(generator),[]);self.assertTrue(cancel.is_set())
        with self.m._cancel_condition:worker=self.m._cancel_worker
        if worker is not None:worker.join(1);self.assertFalse(worker.is_alive())
        self.assertNotIn('aaaaaaaaaaaa',self.m.pending_bindings)
        self.assertNotIn(fg,self.m.operations);self.assertNotIn(rp,self.m.operations)
        self.assertIn('bbbbbbbbbbbb',self.m.bindings);self.assertTrue(e.slot_busy[1])
        self.assertIsNone(self.n.closed);self.assertNotIn('BGEN',e.proc.stdin.text)
        self.assertNotIn('BCANCEL',e.proc.stdin.text)

    def test_cancellation_burst_beyond_four_retains_every_pending_pair(self):
        native=self.n.command;delay=[False];entered=threading.Event();release=threading.Event()
        def blocked(verb,*args,timeout=5.0):
            if delay[0] and verb=='SNAPSHOT':entered.set();release.wait(1)
            out=native(verb,*args)
            if verb=='BIND':out['status']='not_ready'
            return out
        self.n.command=blocked
        requests=['%012x'%value for value in range(1,9)]
        for request in requests:self.assertFalse(self.m.bind(self.t,0,request))
        delay[0]=True
        try:
            self.m.schedule_cancel(requests[0],client=self.n);self.assertTrue(entered.wait(1))
            for request in requests[1:]:self.m.schedule_cancel(request,client=self.n)
            with self.m._cancel_condition:
                self.assertEqual(len(self.m._cancel_pending),8);worker=self.m._cancel_worker
            self.assertIsNone(self.n.closed)
        finally:release.set()
        worker.join(1);self.assertFalse(worker.is_alive())
        self.assertFalse(self.m.pending_bindings);self.assertFalse(self.m.operations)
        self.assertFalse(self.m._cancel_owned);self.assertFalse(self.m._cancel_pending)
        self.assertEqual(sum(row[0]=='RETIRE'for row in self.n.calls),16)
        self.assertIsNone(self.n.closed)

    def test_no_applied_claim_from_accepted_reply(self):
        result=self.m.action({'action':'base','thread':self.t,'base':6,'expected_revision':1})
        self.assertGreater(result['native']['accepted_revision'],result['native']['applied_revision'])
        self.assertEqual(result['native']['applied_cut'],0)


class PumpTests(unittest.TestCase):
    def test_actual_pump_consumes_ready_and_eof_wakes_client(self):
        e=batch_engine(); epoch=e._capture_epoch(); client=SchedulerControl(lambda _:None)
        client.retirements=queue.SimpleQueue();epoch=dataclasses.replace(epoch,scheduler=client);e._epoch=epoch
        e.proc.stdout=io.StringIO('SCHED_READY '+json.dumps(ready())+'\nT 9\n')
        e._pump(epoch)
        self.assertEqual(epoch.lines.get_nowait(),'T 9\n');self.assertIsNone(epoch.lines.get_nowait())
        with self.assertRaises(SchedulerProtocolError):client.wait_ready(.01)
    def test_stale_reply_kills_only_captured_process_and_wakes_waiters(self):
        e=batch_engine();old=e.proc;old.terminate=mock.Mock();epoch=e._capture_epoch()
        client=SchedulerControl(lambda _:None);client.retirements=queue.SimpleQueue()
        epoch=dataclasses.replace(epoch,scheduler=client);e._epoch=epoch
        old.stdout=io.StringIO('SCHED_REPLY {}\n');replace(e)
        e.proc.terminate=mock.Mock();e._pump(epoch)
        old.terminate.assert_called_once();e.proc.terminate.assert_not_called()
        self.assertFalse(e.ended)
    def test_physical_stop_is_written_while_real_graph_reply_unavailable(self):
        from serve.scheduler_control import REPLY_FIELDS
        native=Native(17);entered=threading.Event();release=threading.Event();finished=threading.Event()
        delay=[False]
        def write(frame):
            fields=frame.split();verb=fields[4];kinds=__import__('serve.scheduler_control',fromlist=['GRAMMAR']).GRAMMAR[verb]
            args=[int(value)if kind in ('u','base','bit','boost','ttl')else value
                  for kind,value in zip(kinds,fields[5:])]
            if delay[0] and verb=='SNAPSHOT':entered.set();release.wait(1)
            actual=native.command(verb,*args);record={key:0 for key in REPLY_FIELDS}
            record.update(actual,schema=1,engine_epoch=17,sequence=int(fields[3]),actor_id='',next_expiry_ms=None)
            client.observe('SCHED_REPLY '+json.dumps(record)+'\n')
            if verb=='CANCEL':finished.set()
        client=SchedulerControl(write);client.retirements=queue.SimpleQueue()
        client.observe('SCHED_READY '+json.dumps(ready())+'\n')
        with tempfile.TemporaryDirectory()as directory:
            manager=SchedulerManager(Path(directory)/'state.json');manager.attach(client)
            thread=manager.ensure_default(3);manager.bind(thread,0,'aaaaaaaaaaaa')
            e=batch_engine();epoch=dataclasses.replace(e._capture_epoch(),scheduler=client);e._epoch=epoch
            e.scheduler_manager=manager;delay[0]=True
            try:
                e._send_owned_stop(epoch,0,'aaaaaaaaaaaa')
                self.assertEqual(e.proc.stdin.text,'BSTOP 0\n')
                self.assertTrue(entered.wait(1));self.assertFalse(finished.is_set())
            finally:release.set()
            self.assertTrue(finished.wait(1))

    def test_uncertain_command_failure_callback_owns_original_epoch(self):
        old=mock.Mock();new=mock.Mock();current=[old]
        client=SchedulerControl(lambda _:None,on_failure=lambda _:old.terminate())
        client.observe('SCHED_READY '+json.dumps(ready())+'\n');current[0]=new
        with self.assertRaises(SchedulerProtocolError):client.command('CREATE','thread',1,timeout=.001)
        old.terminate.assert_called_once();new.terminate.assert_not_called()

if __name__=='__main__':unittest.main()

"""Bounded graph revision retries and captured-epoch transport failures."""
import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from serve.scheduler_manager import SchedulerManager,SchedulerManagerError,SchedulerGraphConflict
from serve.scheduler_control import SchedulerProtocolError
class Native:
 def __init__(self,statuses):self.statuses=iter(statuses);self.calls=[];self.closed=False;self.hook=None
 def command(self,v,*a,timeout=5):
  self.calls.append((v,a,timeout))
  if self.hook:self.hook(v)
  r=dict(engine_epoch=7,thread_id='t',thread_generation=4,graph_revision=len(self.calls),status='ok')
  if v!='SNAPSHOT':
   value=next(self.statuses)
   if isinstance(value,Exception):raise value
   r['status']=value;r['operation_serial']=a[1];r['result_revision']=a[2]
  return r
 def close(self,r):self.closed=True
class Tests(unittest.TestCase):
 def setup(self,statuses):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
  m=SchedulerManager(Path(tmp.name)/'s');n=Native(statuses);m.client=n
  m.native={'t':dict(engine_epoch=7,thread_generation=4,graph_revision=0)}
  m.operations={'o':{'thread':'t','reply':dict(operation_serial=8,result_revision=9)},'p':{'thread':'t','reply':dict(operation_serial=10,result_revision=11)}}
  return m,n
 def test_once(self):
  m,n=self.setup(['ok']);m.action(dict(action='cancel',operation='o'));self.assertEqual(len(n.calls),2)
 def test_retry_same_intent_and_local_stale_snapshot(self):
  m,n=self.setup(['stale_version','ok'])
  n.hook=lambda v:self.assertEqual(m.native['t']['graph_revision'],0)
  m.action(dict(action='cancel',operation='o'));self.assertEqual([c[1][1:] for c in n.calls if c[0]=='CANCEL'],[(8,9),(8,9)])
 def test_bounded_conflict_retains_handle_cache_and_health(self):
  m,n=self.setup(['stale_version']*2)
  with self.assertRaises(SchedulerGraphConflict) as caught:m.action(dict(action='retire',operation='o'))
  self.assertEqual((caught.exception.status,caught.exception.attempts),('stale_version',2))
  self.assertEqual(len(n.calls),4);self.assertIn('o',m.operations);self.assertFalse(n.closed);self.assertEqual(m.native['t']['graph_revision'],0)
 def test_other_errors_never_retry(self):
  for error in ['cycle','not_owned','terminal','unknown_generation','error',SchedulerProtocolError('timeout')]:
   m,n=self.setup([error])
   with self.assertRaises((SchedulerManagerError,SchedulerProtocolError)):m.action(dict(action='cancel',operation='o'))
   self.assertEqual(len(n.calls),2)
 def test_retire_only_accepted(self):
  m,n=self.setup(['stale_version','ok']);m.action(dict(action='retire',operation='o'));self.assertNotIn('o',m.operations)
 def test_edge_declarative_exact_endpoints(self):
  m,n=self.setup(['stale_version','ok']);m.action(dict(action='edge',consumer='o',prerequisite='p',add=False));self.assertEqual([c[1][1:] for c in n.calls if c[0]=='EDGE'],[(8,9,10,11,0)]*2)
 def test_generation_mismatch_no_mutation(self):
  m,n=self.setup(['ok']);m.native['t']['thread_generation']=5
  with self.assertRaises(SchedulerProtocolError):m.action(dict(action='cancel',operation='o'))
  self.assertEqual(len(n.calls),1)
 def test_replacement_never_written_or_closed(self):
  m,n=self.setup(['ok']);replacement=Native([])
  n.hook=lambda v:setattr(m,'client',replacement)
  with self.assertRaises(SchedulerManagerError):m.action(dict(action='cancel',operation='o'))
  self.assertEqual(len(n.calls),1);self.assertFalse(replacement.closed);self.assertFalse(replacement.calls)
 def test_changed_epoch_no_cache_commit(self):
  m,n=self.setup(['ok']);m.native['t']['engine_epoch']=8
  with self.assertRaises(SchedulerProtocolError):m.action(dict(action='cancel',operation='o'))
  self.assertEqual(len(n.calls),1);self.assertTrue(n.closed)
 def test_wrong_result_identity_quarantines_before_cache_commit(self):
  m,n=self.setup(['ok']);command=n.command
  def wrong(v,*args,**kwargs):
   r=command(v,*args,**kwargs)
   if v=='CANCEL':r['operation_serial']=999
   return r
  n.command=wrong
  with self.assertRaises(SchedulerProtocolError):m.action(dict(action='cancel',operation='o'))
  self.assertTrue(n.closed);self.assertEqual(m.native['t']['graph_revision'],0)
 def test_deadline_actual_wire_remaining(self):
  m,n=self.setup(['stale_version','ok'])
  with patch('serve.scheduler_manager.time.monotonic',side_effect=[0,0,4,8,11]):
   with self.assertRaises(SchedulerManagerError):m.action(dict(action='cancel',operation='o'))
  self.assertEqual([c[2] for c in n.calls],[5,5,2]);self.assertEqual(len(n.calls),3)

class Transport(unittest.TestCase):
 def client(self,writer):
  import json
  from serve.scheduler_control import SchedulerControl
  from serve.test_scheduler_control import READY
  c=SchedulerControl(writer);c.observe('SCHED_READY '+json.dumps(READY)+'\n');return c
 def test_serial_timeout_quarantines_without_write(self):
  frames=[];c=self.client(frames.append);c._serial.acquire()
  try:
   with self.assertRaises(SchedulerProtocolError):c.command('SNAPSHOT','thread-a',1,timeout=.001)
  finally:c._serial.release()
  self.assertFalse(frames)
  with self.assertRaises(SchedulerProtocolError):c.command('SNAPSHOT','thread-a',1)
 def test_late_immediate_reply_cannot_succeed_after_write_deadline(self):
  from serve.test_scheduler_control import reply
  clock=[0]
  def write(frame):clock[0]=2;c.observe(reply())
  c=self.client(write)
  with patch('serve.scheduler_control.time.monotonic',side_effect=lambda:clock[0]):
   with self.assertRaises(SchedulerProtocolError):c.command('SNAPSHOT','thread-a',1,timeout=1)
  with self.assertRaises(SchedulerProtocolError):c.command('SNAPSHOT','thread-a',1)

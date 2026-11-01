"""Synthetic protocol controls only. Never native numerical/ownership proof."""
import copy,json,unittest
import read_two_slot as r
import lifecycle
PRODUCER=dict(pid=123,start_ticks=99,epoch='e',exe_sha256='a'*64,argv=['engine'])
SEED='111111111111';A='aaaaaaaaaaaa';B='bbbbbbbbbbbb'

def cut(n,f,h,wm,exact=None):return dict(cut=n,forwarded=f,accepted=len(h),exact=f if exact is None else exact,watermark=wm,history=r.digest(h))
def fixture(abort=False,until_seal=False):
 lines=[];seed=[10,11,12,13];old=seed+[90,91,92];ids=[20,21,22,23]+old[4:]+[101,102,103,104];bids=[30,31,32,33]
 expected={A:dict(ids=ids,max_new=64,donor_request=SEED,edit_span=[0,4]),B:dict(ids=bids,max_new=64)};donors={SEED:dict(ids=seed,max_new=3)}
 def emit(kind,**row):lines.append(kind+' '+json.dumps(row))
 def accept(key,leg,slot,occ,prompt,history,f,wm,n,budget):
  emit('BOWN_ACCEPT',request=key,leg=leg,slot=slot,occupancy=occ,full_budget=budget,input_count=len(prompt),input_sha256=r.digest(prompt),cut=n,forwarded=f,accepted=len(history),watermark=wm,history=r.digest(history))
 def http(leg,prompt):emit('HTTP_ACCEPT',request=leg,input_count=len(prompt),input_sha256=r.digest(prompt),**r.serial.NEUTRAL)
 def group(op,entries):emit('BOWN_GROUP',op=op,stages=2,active_rows=len(entries),rows=[dict(slot=s,occupancy=o,position=p)for s,o,p in entries])
 def commit(key,slot,occ,op,n,h,wm):
  lines.append('BT %d %d'%(slot,h[-1]));emit('BOWN_COMMIT',request=key,slot=slot,occupancy=occ,op=op,cut=n,forwarded=len(h)-1,accepted=len(h),watermark=wm,history=r.digest(h),sample=h[-1])
 http(1,seed);lines.append('T 90');accept(SEED,1,0,1,seed,seed+[90],4,0,1,3)
 for op,sample in [(1,91),(2,92)]:group(op,[(0,1,3+op)]);commit(SEED,0,1,op,op+1,seed+[90,91,92][:op+1],op)
 emit('BOWN_DONE',request=SEED,slot=0,occupancy=1,reason='length',watermark=2,history=r.digest(old))
 bh=bids+[40];http(2,bids);lines.append('T 40');accept(B,2,1,2,bids,bh,4,0,1,64)
 http(3,ids);current=cut(2,6,old,2);prefix=cut(50,0,[],0)
 emit('BPREALIGN_BEGIN',request=A,leg=3,slot=0,occupancy=3,full_budget=64,input_count=len(ids),input_sha256=r.digest(ids),page=4,planned_u=8,source=cut(3,6,old,2),installed=current,prefix=prefix)
 ah=list(old)
 for index in range(2):
  op=3+index;group(op,[(0,3,current['forwarded']),(1,2,len(bh)-1)])
  prior=len(ah);actual_input=ah[-1];ah.append(ids[prior]);current=cut(current['cut']+1,len(ah)-1,ah,2)
  emit('BPREALIGN_KNOWN',request=A,slot=0,occupancy=3,op=op,position=current['forwarded']-1,input=actual_input,next_known=ah[-1],actual=current)
  bh.append(41+index);commit(B,1,2,op,index+2,bh,index+1)
  if abort:
   emit('BPREALIGN_ABORT',request=A,slot=0,occupancy=3,last_op=op,reason='cancel',phase='donor',retired=current)
   group(op+1,[(1,2,len(bh)-1)]);bh.append(42);commit(B,1,2,op+1,3,bh,2)
   return expected,donors,lines
 emit('BPREALIGN_SEAL',request=A,slot=0,occupancy=3,captured=current)
 if until_seal:return expected,donors,lines
 ah=ids[:9];accept(A,3,0,3,ids,ah,8,2,100,64)
 for op in (5,6):
  group(op,[(0,3,len(ah)-1),(1,2,len(bh)-1)])
  ah.append(ids[len(ah)]);emit('BOWN_KNOWN',request=A,slot=0,occupancy=3,op=op,cut=96+op,forwarded=len(ah)-1,accepted=len(ah),watermark=2,history=r.digest(ah),next_known=ah[-1])
  bh.append(38+op);commit(B,1,2,op,op-1,bh,op-2)
 group(7,[(0,3,len(ah)-1),(1,2,len(bh)-1)])
 ah.append(105);commit(A,0,3,7,103,ah,3);bh.append(45);commit(B,1,2,7,6,bh,5)
 for key,slot,occ,h,wm in [(A,0,3,ah,3),(B,1,2,bh,5)]:emit('BOWN_DONE',request=key,slot=slot,occupancy=occ,reason='stop',watermark=wm,history=r.digest(h))
 return expected,donors,lines

def parse(case):
 expected,donors,lines=case;records=[dict(kind='producer',schema=1,**PRODUCER)]+[dict(kind='native',line=x)for x in lines]
 return r.validate(('\n'.join(map(json.dumps,records))+'\n').encode(),PRODUCER,expected,donor_requests=donors,sealed=False)

class Pending(unittest.TestCase):
 def mutate(self,prefix,key,value):
  case=fixture();lines=case[2];i=next(i for i,x in enumerate(lines)if x.startswith(prefix+' '));d=json.loads(lines[i].split(' ',1)[1]);d[key]=value;lines[i]=prefix+' '+json.dumps(d)
  with self.assertRaises(ValueError):parse(case)
 def test_progress_then_accept_and_actual_decode_pair(self):
  result=parse(fixture());self.assertEqual(result['dual_operations'],[7]);self.assertEqual(len(result['prealigned'][A]['rows']),2);self.assertEqual(result['response_bt_counts'][A],1);self.assertFalse(result['unconsumed_rows'])
 def test_pending_mixed_groups_are_not_response_overlap(self):
  result=parse(fixture(until_seal=True));self.assertFalse(result['dual_operations']);self.assertNotIn(A,result['commits']);self.assertIn(0,result['pending_admissions'])
 def test_abort_is_not_response_done_and_peer_can_continue(self):
  result=parse(fixture(abort=True));self.assertIn(A,result['aborted_admissions']);self.assertNotIn(A,result['done']);self.assertNotIn(A,result['commits']);self.assertFalse(result['pending_admissions']);self.assertEqual(result['response_bt_counts'][B],2)
 def test_abort_before_first_row(self):
  case=fixture(until_seal=True);i=next(i for i,x in enumerate(case[2])if x.startswith('BPREALIGN_BEGIN '));begin=json.loads(case[2][i].split(' ',1)[1]);case[2][:]=case[2][:i+1]
  case[2].append('BPREALIGN_ABORT '+json.dumps(dict(request=A,slot=0,occupancy=3,last_op=0,reason='cancel',phase='donor',retired=begin['installed'])))
  result=parse(case);self.assertFalse(result['dual_operations']);self.assertEqual(result['aborted_admissions'][A]['last_op'],0)
 def test_prefix_restore_abort_is_not_old_donor_state(self):
  case=fixture(until_seal=True);begin=json.loads(next(x for x in case[2]if x.startswith('BPREALIGN_BEGIN ')).split(' ',1)[1]);actual=copy.deepcopy(begin['prefix']);actual['cut']=5
  case[2].append('BPREALIGN_ABORT '+json.dumps(dict(request=A,slot=0,occupancy=3,last_op=4,reason='refusal',phase='prefix',retired=actual)))
  self.assertEqual(parse(case)['aborted_admissions'][A]['retired']['watermark'],0)
 def test_imported_cancel_preserves_lineage_base_not_prefix_watermark(self):
  case=fixture(until_seal=True);ids=case[0][A]['ids'];actual=cut(100,8,ids[:9],2,exact=4)
  case[2].append('BPREALIGN_ABORT '+json.dumps(dict(request=A,slot=0,occupancy=3,last_op=4,reason='cancel',phase='imported',retired=actual)))
  result=parse(case);self.assertNotIn(A,result['commits']);self.assertEqual(result['aborted_admissions'][A]['retired']['watermark'],2)
 def test_fullstream_lifecycle_tracks_pending_separately(self):
  result=lifecycle.validate('\n'.join(fixture()[2])+'\n');self.assertEqual(result['native_occupancies'],3);self.assertEqual(result['unpublished_occupancies'],1)
 def test_fullstream_abort_is_not_accepted_occupancy(self):
  case=fixture(abort=True);state=parse(case);case[2].append('BOWN_DONE '+json.dumps(dict(request=B,slot=1,occupancy=2,reason='stop',watermark=2,history=r.digest(state['histories'][B]))))
  result=lifecycle.validate('\n'.join(case[2])+'\n');self.assertEqual(result['native_occupancies'],2);self.assertEqual(result['aborted_unpublished_occupancies'],1)
 def test_fullstream_pending_output_fails(self):
  case=fixture();i=next(i for i,x in enumerate(case[2])if x.startswith('BPREALIGN_KNOWN '));case[2].insert(i,'BT 0 777')
  with self.assertRaises(ValueError):lifecycle.validate('\n'.join(case[2])+'\n')
 def test_fullstream_missing_pending_completion_fails(self):
  case=fixture();i=next(i for i,x in enumerate(case[2])if x.startswith('BPREALIGN_KNOWN '));case[2].pop(i)
  with self.assertRaises(ValueError):lifecycle.validate('\n'.join(case[2])+'\n')
 def test_bad_pending_token(self):self.mutate('BPREALIGN_KNOWN','input',999)
 def test_wrong_known_input(self):self.mutate('BPREALIGN_KNOWN','next_known',999)
 def test_wrong_occupancy(self):self.mutate('BPREALIGN_KNOWN','occupancy',999)
 def test_zero_or_foreign_source_watermark(self):
  case=fixture();i=next(i for i,x in enumerate(case[2])if x.startswith('BPREALIGN_BEGIN '));d=json.loads(case[2][i].split(' ',1)[1]);d['source']['watermark']=0;case[2][i]='BPREALIGN_BEGIN '+json.dumps(d)
  with self.assertRaises(ValueError):parse(case)
 def test_watermark_increment_is_not_output_receipt(self):
  case=fixture();i=next(i for i,x in enumerate(case[2])if x.startswith('BPREALIGN_KNOWN '));d=json.loads(case[2][i].split(' ',1)[1]);d['actual']['watermark']+=1;case[2][i]='BPREALIGN_KNOWN '+json.dumps(d)
  with self.assertRaises(ValueError):parse(case)
 def test_unknown_group_row_without_begin(self):
  case=fixture();case[2][:]=[x for x in case[2]if not x.startswith('BPREALIGN_BEGIN ')]
  with self.assertRaises(ValueError):parse(case)
 def test_unsealed_admission_refuses(self):
  case=fixture();case[2][:]=[x for x in case[2]if not x.startswith('BPREALIGN_SEAL ')]
  with self.assertRaises(ValueError):parse(case)
 def test_duplicate_consumption_refuses(self):
  case=fixture();i=next(i for i,x in enumerate(case[2])if x.startswith('BPREALIGN_KNOWN '));case[2].insert(i+1,case[2][i])
  with self.assertRaises(ValueError):parse(case)
 def test_pending_bt_refuses(self):
  case=fixture();i=next(i for i,x in enumerate(case[2])if x.startswith('BPREALIGN_KNOWN '));case[2].insert(i,'BT 0 777')
  with self.assertRaises(ValueError):parse(case)
 def test_missing_native_row_consumption_cannot_abort(self):
  case=fixture(abort=True);case[2][:]=[x for x in case[2]if not x.startswith('BPREALIGN_KNOWN ')]
  with self.assertRaises(ValueError):parse(case)
 def test_stale_postabort_row_refuses(self):
  case=fixture(abort=True);case[2].append('BOWN_GROUP '+json.dumps(dict(op=99,stages=2,active_rows=1,rows=[dict(slot=0,occupancy=3,position=7)])))
  with self.assertRaises(ValueError):parse(case)
 def test_slice_without_authentic_retired_donor_refuses(self):
  case=fixture();i=next(i for i,x in enumerate(case[2])if x.startswith('HTTP_ACCEPT ')and '"request": 3'in x);case[2][:]=case[2][i:]
  with self.assertRaises(ValueError):parse(case)
if __name__=='__main__':unittest.main()

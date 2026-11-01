"""Actual failed admissions regression; synthetic continuation is never native proof."""
import copy,json,pathlib,unittest
import read_two_slot as r
ROOT=pathlib.Path(__file__).resolve().parent
A='a00000000002';B='b00000000002'
def actual():
 f=json.loads((ROOT/'actual-admissions.json').read_text());return f,(ROOT/'actual-admissions.stdout.log').read_text().splitlines()
def parse(f,lines):
 rows=[dict(kind='producer',schema=1,**f['producer'])]+[dict(kind='native',line=x)for x in lines]
 return r.validate(('\n'.join(map(json.dumps,rows))+'\n').encode(),f['producer'],f['requests'],donor_requests=f['donor_requests'],sealed=False)
def append_progress(f,lines):
 """SYNTHETIC, extends actual admissions to exercise KNOWN and BT accounting."""
 state=parse(f,lines);op=100
 for key in (A,B):
  a=copy.deepcopy(state['admissions'][key]);history=list(state['histories'][key]);ids=f['requests'][key]['ids']
  while len(history)<len(ids):
   op+=1;lines.append('BOWN_GROUP '+json.dumps(dict(op=op,stages=2,active_rows=1,rows=[dict(slot=a['slot'],occupancy=a['occupancy'],position=a['forwarded'])])))
   token=ids[len(history)];history.append(token);a.update(op=op,cut=a['cut']+1,forwarded=a['forwarded']+1,accepted=a['accepted']+1,history=r.digest(history),next_known=token);lines.append('BOWN_KNOWN '+json.dumps(a))
  for sample in (42,43,44):
   op+=1;lines.append('BOWN_GROUP '+json.dumps(dict(op=op,stages=2,active_rows=1,rows=[dict(slot=a['slot'],occupancy=a['occupancy'],position=a['forwarded'])])));lines.append('BT %d %d'%(a['slot'],sample));history.append(sample);a.update(op=op,cut=a['cut']+1,forwarded=a['forwarded']+1,accepted=a['accepted']+1,watermark=a['watermark']+1,history=r.digest(history),sample=sample);a.pop('next_known',None);lines.append('BOWN_COMMIT '+json.dumps(a))
  lines.append('BOWN_DONE '+json.dumps(dict(request=key,slot=a['slot'],occupancy=a['occupancy'],reason='length',watermark=a['watermark'],history=a['history'])))
 return lines
class Lineage(unittest.TestCase):
 def test_unmodified_real_fullraw_admissions_and_real_producer(self):
  f,lines=actual();p=parse(f,lines);self.assertEqual(p['admissions'][A]['watermark'],2);self.assertEqual(p['admissions'][B]['watermark'],1);self.assertEqual(p['admissions'][A]['forwarded'],p['retired']['a00000000001']['forwarded']);self.assertFalse(p['dual_operations'])
 def change(self,prefix,key,value):
  f,lines=actual()
  for i,line in enumerate(lines):
   if line.startswith(prefix) and ('"request": "'+A+'"'in line or '"request":"'+A+'"'in line):
    d=json.loads(line.split(' ',1)[1]);d[key]=value;lines[i]=prefix+json.dumps(d);break
  with self.assertRaises(ValueError):parse(f,lines)
 def test_wrong_or_zero_base_refused(self):
  for value in (0,1,3,False):self.change('BOWN_ACCEPT ','watermark',value)
 def test_wrong_admitted_history(self):self.change('BOWN_ACCEPT ','history','0'*64)
 def test_foreign_real_donor(self):
  f,lines=actual();f['requests'][A]['donor_request']='b00000000001'
  with self.assertRaises(ValueError):parse(f,lines)
 def test_unauthenticated_expectedbase_not_accepted(self):
  f,lines=actual();f['requests'][A].pop('donor_request');f['requests'][A]['expected_watermark']=2
  with self.assertRaises(ValueError):parse(f,lines)
 def test_pair_slice_cannot_authenticate_donor(self):
  f,lines=actual();start=next(i for i,x in enumerate(lines)if x.startswith('HTTP_ACCEPT ')and '"request":3,'in x)
  with self.assertRaises(ValueError):parse(f,lines[start:])
 def test_changed_outside_declared_edit(self):
  f,lines=actual();f['requests'][A]['edit_span']=[12,16]
  with self.assertRaises(ValueError):parse(f,lines)
 def test_late_main_T_refused(self):
  f,lines=actual();lines.append('T 99')
  with self.assertRaises(ValueError):parse(f,lines)
 def test_missing_donor_retirement(self):
  f,lines=actual();lines=[x for x in lines if not(x.startswith('BOWN_DONE ')and 'a00000000001'in x)]
  with self.assertRaises(ValueError):parse(f,lines)
 def test_synthetic_known_preserves_nonzero_base_and_decode_counts_delta(self):
  f,lines=actual();p=parse(f,append_progress(f,lines));self.assertEqual(p['retired'][A]['watermark'],5);self.assertEqual(p['retired'][B]['watermark'],4);self.assertEqual(p['response_bt_counts'][A],3);self.assertEqual(p['response_bt_counts'][B],3)
 def test_synthetic_known_cannot_reset_base(self):
  f,lines=actual();lines=append_progress(f,lines);at=next(i for i,x in enumerate(lines)if x.startswith('BOWN_KNOWN '));d=json.loads(lines[at].split(' ',1)[1]);d['watermark']=0;lines[at]='BOWN_KNOWN '+json.dumps(d)
  with self.assertRaises(ValueError):parse(f,lines)
 def test_synthetic_return_inherits_actual_aggregate_not_response_count(self):
  f,lines=actual();lines=append_progress(f,lines);p=parse(f,lines);prior=p['histories'][A];ids=list(prior);ids[8:12]=[901,902,903,904];ids+=[905]*4
  request='a00000000003';f['donor_requests'].update(f['requests']);f['requests']={request:dict(ids=ids,max_new=64,donor_request=A,edit_span=[8,12]),'unused000001':dict(ids=[1],max_new=2)}
  a=p['retired'][A];lines+=['HTTP_ACCEPT '+json.dumps(dict(request=5,input_count=len(ids),input_sha256=r.digest(ids),**r.serial.NEUTRAL)),'BOWN_ACCEPT '+json.dumps(dict(request=request,slot=0,occupancy=90,leg=5,full_budget=64,input_count=len(ids),input_sha256=r.digest(ids),cut=999,forwarded=a['forwarded'],accepted=a['accepted'],watermark=a['watermark'],history=r.digest(ids[:a['accepted']])))]
  q=parse(f,lines);self.assertEqual(q['admissions'][request]['watermark'],5)
  d=json.loads(lines[-1].split(' ',1)[1]);d['watermark']=3;lines[-1]='BOWN_ACCEPT '+json.dumps(d)
  with self.assertRaises(ValueError):parse(f,lines)
 def test_synthetic_known_row_cannot_consume_same_actual_group_twice(self):
  f,lines=actual();lines=append_progress(f,lines)
  first=next(i for i,x in enumerate(lines)if x.startswith('BOWN_KNOWN '));row=json.loads(lines[first].split(' ',1)[1]);history=list(parse(f,lines[:first+1])['histories'][A]);token=f['requests'][A]['ids'][len(history)];history.append(token);row.update(cut=row['cut']+1,forwarded=row['forwarded']+1,accepted=row['accepted']+1,next_known=token,history=r.digest(history))
  with self.assertRaises(ValueError):parse(f,lines[:first+1]+['BOWN_KNOWN '+json.dumps(row)])
 def test_non_target_donor_group_never_qualifies_target_pair(self):
  f,lines=actual();prefix=[]
  def event(k,**d):prefix.append(k+' '+json.dumps(d))
  ids=[1,2,3,4];donors={}
  for i,key in enumerate(('seedA','seedB')):
   donors[key]=dict(ids=ids,max_new=2);event('HTTP_ACCEPT',request=i+1,input_count=4,input_sha256=r.digest(ids),**r.serial.NEUTRAL);prefix.extend(['T 10','DONE 1 4 1 1 length'])
   event('BOWN_ACCEPT',request=key,slot=i,occupancy=i+1,leg=i+1,full_budget=2,input_count=4,input_sha256=r.digest(ids),cut=1,forwarded=4,accepted=5,watermark=0,history=r.digest(ids+[10]))
  event('BOWN_GROUP',op=1,stages=2,active_rows=2,rows=[dict(slot=i,occupancy=i+1,position=4)for i in (0,1)])
  for i,key in enumerate(donors):
   prefix.append('BT %d 11'%i);event('BOWN_COMMIT',request=key,slot=i,occupancy=i+1,op=1,cut=2,forwarded=5,accepted=6,watermark=1,history=r.digest(ids+[10,11]),sample=11)
  for i,key in enumerate(donors):event('BOWN_DONE',request=key,slot=i,occupancy=i+1,reason='length',watermark=1,history=r.digest(ids+[10,11]))
  f['donor_requests']=donors;f['requests']={'targetA':dict(ids=[9],max_new=2),'targetB':dict(ids=[8],max_new=2)}
  self.assertFalse(parse(f,prefix)['dual_operations'])
if __name__=='__main__':unittest.main()

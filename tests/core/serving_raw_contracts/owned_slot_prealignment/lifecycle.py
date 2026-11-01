"""Full stdout lifecycle guard around sliced numerical/history comparisons."""
import importlib.util
from pathlib import Path
_reader=Path(__file__).resolve().with_name('read_two_slot.py')
if not _reader.exists():_reader=Path(__file__).resolve().parents[1]/'http-two-slot-qualification-sol/read_two_slot.py'
_spec=importlib.util.spec_from_file_location('return_full_lifecycle_reader',_reader)
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
strict=_module.strict
need=_module.need

def validate(text):
 need(type(text) is str and len(text.encode())<=2<<20,'Bounded full native stdout required')
 aligning={};seen_pending=set();sealed_pending=set();aborted=set()
 active={};pending={};stopped=set();groups={};last_op=0;seen=set();consumed=set();leg=None;leg_open=False;leg_tokens=0;seen_legs=set()
 for line in text.splitlines():
  need(not line.startswith(('ERR','BOWN_ERROR','BPREALIGN_ERROR')),'Native refusal in full transcript')
  if line.startswith('HTTP_ACCEPT '):
   row=strict(line.split(' ',1)[1]);ordinal=row['request']
   need(leg is None and type(ordinal) is int and ordinal>0 and ordinal not in seen_legs,'Overlapping/repeated actual main admission')
   seen_legs.add(ordinal);leg=ordinal;leg_open=True;leg_tokens=0
  elif line.startswith('T '):
   fields=line.split();need(leg is not None and leg_open and len(fields)==2 and int(fields[1])>=0,'T outside actual open main admission');leg_tokens+=1
  elif line.startswith('DONE '):
   fields=line.split();need(leg is not None and leg_open and len(fields)>=6 and int(fields[1])==leg_tokens and fields[5] in ('length','stop'),'DONE outside actual open main admission');leg_open=False
  if line.startswith('BPREALIGN_'):
   kind,raw=line.split(' ',1);row=strict(raw);slot=row['slot'];identity=(row['request'],row['occupancy']);physical=(slot,row['occupancy'])
   if kind=='BPREALIGN_BEGIN':
    need(type(slot) is int and slot in (0,1) and type(row['occupancy']) is int and row['occupancy']>0 and slot not in active and slot not in aligning and physical not in seen and physical not in seen_pending,'Repeated/unowned pending occupancy')
    need(leg is not None and leg_open and not leg_tokens and row['leg']==leg,'Pending occupancy lacks actual open main admission')
    aligning[slot]=identity;seen_pending.add(physical)
   else:
    need(aligning.get(slot)==identity,'Pending event after retirement/on foreign occupancy')
    if kind=='BPREALIGN_KNOWN':
     need(physical not in sealed_pending and groups.get(row['op'],{}).get(slot)==row['occupancy'] and (slot,row['op']) not in consumed and slot not in pending,'Pending known lacks real unused group or emits BT')
     consumed.add((slot,row['op']))
    elif kind=='BPREALIGN_SEAL':
     need(physical not in sealed_pending and all((slot,op) in consumed for op,rows in groups.items() if rows.get(slot)==row['occupancy']),'Pending seal before all real native rows complete')
     sealed_pending.add(physical)
    elif kind=='BPREALIGN_ABORT':
     need(all((slot,op) in consumed for op,rows in groups.items() if rows.get(slot)==row['occupancy']) and slot not in pending,'Pending abort before real drain/output settlement')
     need(leg is not None and leg_open,'Pending abort lacks actual main admission')
     aligning.pop(slot);aborted.add(identity);leg=None;leg_open=False
    else:need(False,'Unknown pending lifecycle event')
  if line.startswith('BOWN_'):
   kind,raw=line.split(' ',1);row=strict(raw)
   if kind=='BOWN_ACCEPT':
    slot=row['slot'];key=row['request'];occ=row['occupancy']
    need(type(slot) is int and slot in (0,1) and type(occ) is int and occ>0 and slot not in active and (slot,occ) not in seen,'Repeated/foreign live occupancy')
    if slot in aligning:
     need(aligning[slot]==(key,occ) and (slot,occ) in sealed_pending,'Pending occupancy accepted before genuine seal');aligning.pop(slot)
    else:need((slot,occ) not in seen_pending,'Retired unpublished occupancy reused as response')
    need(leg is not None and row['leg']==leg,'Slot acceptance lacks its actual main admission leg')
    leg=None;leg_open=False
    seen.add((slot,occ));active[slot]=(key,occ)
   elif kind=='BOWN_GROUP':
    op=row['op'];rows=row['rows']
    need(type(op) is int and op>last_op and row['active_rows']==len(rows) and 1<=len(rows)<=2 and len({x['slot'] for x in rows})==len(rows),'Malformed/replayed actual group')
    for entry in rows:
     current=active.get(entry['slot']) or aligning.get(entry['slot']);need(current and current[1]==entry['occupancy'] and current not in stopped and current not in aborted,'Group after retirement/stop or wrong occupancy')
     need((entry['slot'],entry['occupancy']) not in sealed_pending or entry['slot'] in active,'Native work after unpublished seal')
    groups[op]={x['slot']:x['occupancy'] for x in rows};last_op=op
   elif kind in ('BOWN_COMMIT','BOWN_KNOWN','BOWN_STOP','BOWN_DONE'):
    slot=row['slot'];current=active.get(slot);identity=(row['request'],row['occupancy'])
    need(current==identity,'Native event after retirement or on foreign occupancy')
    if kind in ('BOWN_COMMIT','BOWN_KNOWN'):
     need(current not in stopped and groups.get(row['op'],{}).get(slot)==row['occupancy'],'Commit/known row after stop or without actual group')
     need((slot,row['op']) not in consumed,'Repeated group row completion');consumed.add((slot,row['op']))
     if kind=='BOWN_COMMIT':need(pending.pop(slot,None)==row['sample'],'Commit lacks one actual unused BT')
     else:need(slot not in pending,'Known row emitted a token')
    elif kind=='BOWN_STOP':need(current not in stopped and slot not in pending,'Repeated stop or incomplete output publication');stopped.add(current)
    else:need(slot not in pending,'Retirement with unmatched output');active.pop(slot)
  elif line.startswith('BT '):
   fields=line.split();need(len(fields)==3,'Malformed native BT');slot=int(fields[1]);sample=int(fields[2]);current=active.get(slot)
   need(current and current not in stopped and slot not in pending and sample>=0,'BT after retirement/stop or duplicate uncommitted output');pending[slot]=sample
  # PP is observation only; slot output always follows actual occupancy.
 need(leg is None and not leg_open,'Final process left an unresolved main admission')
 need(all((slot,op) in consumed for op,rows in groups.items() for slot in rows),'Actual group row lacks exactly one completion')
 need(not active and not pending and not aligning,'Final native process left live slot output/occupancy')
 return {'complete_stdout_lifecycle_validated':True,'native_occupancies':len(seen),'unpublished_occupancies':len(seen_pending),'aborted_unpublished_occupancies':len(aborted)}

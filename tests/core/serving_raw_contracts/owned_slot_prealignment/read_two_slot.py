"""Strict native batch2 observations. No standalone deployment qualification."""
import importlib.util,json,pathlib,sys
P=pathlib.Path(__file__).with_name('read_receipts.py')
if not P.is_file():P=pathlib.Path(__file__).parents[1]/'http-native-evidence-sol/read_receipts.py'
spec=importlib.util.spec_from_file_location('serial_native_reader',P);serial=importlib.util.module_from_spec(spec);spec.loader.exec_module(serial)
need=serial.need;strict=serial.strict;digest=serial.history_digest

def validate(raw,producer,requests,*,sealed=True,donor_requests=None,admission_requests=None):
    need(type(raw) is bytes and len(raw)<=2<<20 and raw.endswith(b'\n'),'Incomplete/oversized native observations')
    records=[strict(x) for x in raw.splitlines()]
    need(records and records[0]=={'kind':'producer','schema':1,**producer},'Different actual native producer')
    need(set(producer)=={'pid','start_ticks','epoch','exe_sha256','argv'} and type(producer['pid']) is int and producer['pid']>1,'Explicit process identity required')
    if sealed:
        need(records[-1]=={'kind':'eof','exit_code':0},'Actual native EOF0 required');records=records[1:-1]
    else:records=records[1:]
    need(len(requests)==2 and len(set(requests))==2,'Two distinct actual HTTP request IDs required')
    donor_requests={} if donor_requests is None else donor_requests
    need(type(donor_requests) is dict and not (set(donor_requests)&set(requests)),'Disjoint actual donor request map required')
    all_requests={**donor_requests,**requests}
    admission_requests={} if admission_requests is None else admission_requests
    need(type(admission_requests) is dict and not(set(admission_requests)&set(all_requests)),'Disjoint pending-only request map required')
    all_inputs={**all_requests,**admission_requests}
    aligning={};alignment_done={};aborted={};row_inflight={};seen_occupancies=set()
    def checked_cut(c):
        need(type(c) is dict and set(c)=={'cut','forwarded','accepted','exact','watermark','history'} and
             all(type(c[k]) is int and c[k]>=0 for k in ('cut','forwarded','accepted','exact','watermark')) and
             c['cut']>0 and c['exact']<=c['forwarded']<=c['accepted'],'Invalid prealignment actual cut')
        need(type(c['history']) is str and len(c['history'])==64 and all(x in '0123456789abcdef' for x in c['history']),'Invalid prealignment history digest')
        return c
    def same_endpoint(a,b):return all(a[k]==b[k] for k in ('forwarded','accepted','watermark','history'))
    active={};groups={};pending={};commits={};stops={};done={};dual=[];last_op=0;legs={};leg=None;histories={};admissions={};retired={};consumed=set()
    for event in records:
        need(set(event)=={'kind','line'} and event['kind']=='native','Unknown observation')
        line=event['line']
        need(not line.startswith(('ERR','BOWN_ERROR','BPREALIGN_ERROR')),'Actual native refusal')
        if line.startswith('HTTP_ACCEPT '):
            row=strict(line.split(' ',1)[1]);leg=row['request']
            need(all(row.get(k)==v for k,v in serial.NEUTRAL.items()),'Actual native sampling is not neutral')
            need(leg not in legs,'Repeated native admission leg');legs[leg]={'accepted':row,'tokens':[]}
        elif line.startswith('T '):
            parts=line.split();need(leg in legs and len(parts)==2 and int(parts[1])>=0,'Admission T lacks actual accepted leg/token');legs[leg]['tokens'].append(int(parts[1]))
        if line.startswith('BPREALIGN_'):
            kind,rawrow=line.split(' ',1);r=strict(rawrow)
            if kind=='BPREALIGN_BEGIN':
                need(set(r)=={'request','leg','slot','occupancy','full_budget','input_count','input_sha256','page','planned_u','source','installed','prefix'},'Unknown pending admission fields')
                key=r['request'];slot=r['slot'];occ=r['occupancy'];need(key in all_inputs and key not in commits and key not in aborted and key not in alignment_done and
                    slot in (0,1) and slot not in active and slot not in aligning and type(occ) is int and occ>0 and (slot,occ) not in seen_occupancies,'Repeated/stale prealignment occupancy')
                expected=all_inputs[key];ids=expected['ids'];donor_key=expected.get('donor_request');donor=retired.get(donor_key)
                need(donor is not None and done[donor_key]['reason'] in ('length','stop'),'Pending admission lacks genuine retired donor')
                span=expected.get('edit_span');need(type(span) is list and len(span)==2 and all(type(x) is int for x in span),'Pending admission lacks actual edit span')
                lo,hi=span;prior=histories[donor_key]
                need(0<=lo<hi<=donor['forwarded'] and lo%4==hi%4==0 and len(ids)>len(prior) and ids[:lo]==prior[:lo] and
                    ids[hi:len(prior)]==prior[hi:] and ids[lo]!=prior[lo] and ids[hi-1]!=prior[hi-1],'Pending request differs outside genuine donor edit')
                native_leg=legs.get(r['leg']);need(native_leg and not native_leg['tokens'] and native_leg['accepted']['input_count']==len(ids) and
                    native_leg['accepted']['input_sha256']==digest(ids),'Pending admission lacks genuine validated leg')
                need(r['full_budget']==expected['max_new'] and r['input_count']==len(ids) and r['input_sha256']==digest(ids),'Pending input/budget differs')
                source=checked_cut(r['source']);current=checked_cut(r['installed']);prefix=checked_cut(r['prefix'])
                need(same_endpoint(source,donor) and source['exact']==source['forwarded'] and source['accepted']==source['forwarded']+1 and
                    current['exact']==source['exact'] and same_endpoint(current,source),'Installed donor is not actual retired source endpoint')
                u=r['planned_u'];need(type(u) is int and r['page']==4 and source['forwarded']%4!=0 and u==((source['forwarded']+3)//4)*4 and
                    1<=u-source['forwarded']<=3 and u<len(ids),'Pending admission plan has no bounded genuine next-known tail')
                need(prefix['accepted']==prefix['forwarded']==prefix['exact'] and prefix['forwarded']<=lo and
                    prefix['history']==digest(ids[:prefix['accepted']]),'Prefix operand is not genuine exact incoming prefix')
                aligning[slot]={'request':key,'occupancy':occ,'begin':r,'cut':dict(current),'history':list(prior),'last_op':0,'sealed':False,'rows':[]}
                seen_occupancies.add((slot,occ))
            else:
                slot=r['slot'];a=aligning.get(slot);need(type(slot) is int and type(r['occupancy']) is int and a and a['request']==r['request'] and a['occupancy']==r['occupancy'],'Pending event uses foreign/retired occupancy')
                ids=all_inputs[a['request']]['ids'];before=a['cut'];receipt=(r.get('op'),slot,r['occupancy'])
                if kind=='BPREALIGN_KNOWN':
                    need(set(r)=={'request','slot','occupancy','op','position','input','next_known','actual'} and all(type(r[k]) is int and r[k]>=0 for k in ('op','position','input','next_known')) and not a['sealed'],'Known row after seal or unknown fields')
                    group=groups.get(r['op']);row=next((x for x in group['rows'] if x['slot']==slot and x['occupancy']==r['occupancy']),None) if group else None
                    need(row and row_inflight.get((slot,r['occupancy']))==r['op'] and receipt not in consumed and
                        row['position']==r['position']==before['forwarded'] and r['input']==a['history'][-1],'Pending row lacks authentic unconsumed group/input')
                    need(before['forwarded']<a['begin']['planned_u'] and r['next_known']==ids[before['accepted']],'Pending seal skipped or invented known input')
                    actual=checked_cut(r['actual']);a['history'].append(r['next_known'])
                    need(actual['cut']>before['cut'] and actual['forwarded']==before['forwarded']+1 and actual['accepted']==before['accepted']+1 and
                        actual['exact']==actual['forwarded'] and actual['watermark']==before['watermark'] and actual['history']==digest(a['history']),'Pending known cut/history/output watermark differs')
                    consumed.add(receipt);row_inflight.pop((slot,r['occupancy']));a['cut']=dict(actual);a['last_op']=r['op'];a['rows'].append(r)
                elif kind=='BPREALIGN_SEAL':
                    need(set(r)=={'request','slot','occupancy','captured'} and not a['sealed'] and (slot,r['occupancy']) not in row_inflight,'Repeated seal or outstanding native row')
                    captured=checked_cut(r['captured']);need(captured==before and captured['forwarded']==a['begin']['planned_u'] and
                        len(a['rows'])==captured['forwarded']-a['begin']['source']['forwarded'],'Seal is not real fully progressed aligned endpoint')
                    a['sealed']=True
                elif kind=='BPREALIGN_ABORT':
                    need(set(r)=={'request','slot','occupancy','last_op','reason','phase','retired'} and r['reason'] in ('cancel','refusal','shutdown') and
                        r['last_op']==a['last_op'] and (slot,r['occupancy']) not in row_inflight,'Abort before native drain or wrong operation/reason')
                    actual=checked_cut(r['retired']);phase=r['phase']
                    if phase=='donor':need(actual==before,'Donor abort differs from actual pending endpoint')
                    elif phase=='prefix':need(a['sealed'] and same_endpoint(actual,a['begin']['prefix']) and actual['exact']==actual['forwarded'] and actual['cut']>before['cut'],'Prefix abort differs from actual restored operand')
                    elif phase=='imported':need(a['sealed'] and actual['forwarded']==a['begin']['planned_u'] and actual['accepted']==actual['forwarded']+1 and
                        actual['watermark']==a['begin']['source']['watermark'] and actual['history']==digest(ids[:actual['accepted']]) and
                        a['begin']['prefix']['exact']<=actual['exact']<=actual['forwarded'] and actual['cut']>before['cut'],'Imported abort differs from actual no-output request history')
                    else:need(False,'Unknown actual abort phase')
                    aborted[a['request']]=r;aligning.pop(slot)
                else:need(False,'Unknown prealignment record')
        if line.startswith('BOWN_'):
            kind,rawrow=line.split(' ',1);r=strict(rawrow)
            if kind=='BOWN_ACCEPT':
                request=r['request'];slot=r['slot'];occ=r['occupancy']
                need(request in all_requests and request not in commits and slot in (0,1) and slot not in active and type(occ) is int and occ>0,'Unexpected native slot admission')
                pre=aligning.get(slot)
                need((pre is None and (slot,occ) not in seen_occupancies) or (pre and pre['request']==request and pre['occupancy']==occ and pre['sealed'] and (slot,occ) not in row_inflight),'Accepted pending occupancy was not genuinely sealed')
                expected=all_requests[request]
                need(r['input_count']==len(expected['ids']) and r['input_sha256']==digest(expected['ids']) and r['full_budget']==expected['max_new'],'Native input/full budget differs')
                native_leg=legs.get(r['leg']);need(native_leg is not None,'Owned admission lacks native validated leg')
                need(native_leg['accepted']['input_sha256']==r['input_sha256'] and native_leg['accepted']['input_count']==r['input_count'],'Native validated leg differs')
                need(type(r['forwarded']) is int and 0<=r['forwarded']<=len(expected['ids']) and r['accepted']==r['forwarded']+1 and type(r['watermark']) is int and r['watermark']>=0,'Invalid actual initial slot cut/watermark')
                emitted=native_leg['tokens']
                # Main admission T is real history, not a slot BT watermark.
                # Full ordinary admission emitted exactly its pending sample; early repaired
                # admission holds an actual validated input prefix and emitted none.
                if r['forwarded']==len(expected['ids']):
                    need(len(emitted)==1 and r['watermark']==0 and 'donor_request' not in expected,'Ordinary admission requires main sample and fresh slot watermark')
                else:
                    donor_key=expected.get('donor_request');donor=retired.get(donor_key)
                    need(not emitted and donor is not None and donor_key!=request,'Repaired admission needs actual retired donor, not an arbitrary watermark base')
                    need(done[donor_key]['reason'] in ('length','stop'),'Canceled/invalid donor cannot authenticate repaired admission')
                    span=expected.get('edit_span');need(type(span) is list and len(span)==2 and all(type(x) is int for x in span),'Actual aligned donor edit span required')
                    lo,hi=span;prior=histories[donor_key];ids=expected['ids']
                    need(0<=lo<hi<=donor['forwarded'] and lo%4==hi%4==0 and len(ids)>len(prior) and
                         ids[:lo]==prior[:lo] and ids[hi:len(prior)]==prior[hi:] and
                         ids[lo]!=prior[lo] and ids[hi-1]!=prior[hi-1], 'Repaired prompt is not the genuine same-length edit of actual donor history plus fresh tail')
                    inherited=pre['cut'] if pre else donor
                    need((pre is None or pre['begin']['source']['history']==donor['history']) and
                         r['forwarded']==inherited['forwarded'] and r['accepted']==inherited['accepted'] and
                         r['watermark']==donor['watermark'],'Repaired admission differs from actual donor/progressed terminal cut')
                    if pre:need(r['cut']>pre['cut']['cut'],'Imported request did not issue a new actual cut')
                history=(expected['ids']+emitted)[:r['accepted']]
                need(len(history)==r['accepted'] and digest(history)==r['history'],'Actual admitted history differs')
                if pre:alignment_done[request]=pre;aligning.pop(slot)
                seen_occupancies.add((slot,occ))
                histories[request]=history;admissions[request]=dict(r)
                active[slot]=dict(r);commits[request]=[];leg=None
            elif kind=='BOWN_GROUP':
                op=r['op'];rows=r['rows']
                need(type(op) is int and op>last_op and r['stages']==2 and r['active_rows']==len(rows) and 1<=len(rows)<=2,'Native group identity/stages invalid')
                need(len({x['slot'] for x in rows})==len(rows),'Duplicate native slot row')
                for row in rows:
                    a=active.get(row['slot']) or aligning.get(row['slot']);need(a and a['occupancy']==row['occupancy'] and a['request'] not in stops and a['request'] not in done,'Group uses stopped/stale occupancy')
                    position=a['cut']['forwarded'] if row['slot'] in aligning else a['forwarded']
                    need(row['position']==position and (row['slot'],row['occupancy']) not in row_inflight,'Actual row position differs or previous receipt remains unconsumed')
                    if row['slot'] in aligning:need(not a['sealed'],'Native work after pending seal')
                    row_inflight[(row['slot'],row['occupancy'])]=op
                groups[op]=r;last_op=op
            elif kind in ('BOWN_COMMIT','BOWN_KNOWN'):
                a=active.get(r['slot']);request=r['request'];op=r['op']
                need(a and a['request']==request and a['occupancy']==r['occupancy'] and request not in stops and request not in done,'Commit uses stopped/foreign occupancy')
                g=groups.get(op);row=next((x for x in g['rows'] if x['slot']==r['slot'] and x['occupancy']==r['occupancy']),None) if g else None
                receipt_key=(op,r['slot'],r['occupancy'])
                need(row is not None and row['position']==a['forwarded'] and receipt_key not in consumed and row_inflight.get((r['slot'],r['occupancy']))==op,'Commit/known reuses a consumed native row or wrong actual position')
                known=kind=='BOWN_KNOWN'
                if known:
                    need(r['slot'] not in pending and a['watermark']==admissions[request]['watermark'] and not commits[request] and a['accepted']<len(all_requests[request]['ids']),'Known row after decode or beyond input')
                    need(r['next_known']==all_requests[request]['ids'][a['accepted']],'Known row differs from actual validated input')
                    histories[request].append(r['next_known'])
                else:
                    need(pending.pop(r['slot'],None)==r['sample'],'Commit lacks matching genuine BT')
                    need(a['accepted']>=len(all_requests[request]['ids']),'Decode before full known input')
                    histories[request].append(r['sample'])
                need(r['forwarded']==a['forwarded']+1 and r['accepted']==a['accepted']+1 and r['watermark']==a['watermark']+(0 if known else 1) and r['cut']>a['cut'],'Nonmonotonic actual cut')
                need(digest(histories[request])==r['history'],'Actual committed history differs')
                need(not commits[request] or op>commits[request][-1]['op'],'Replayed group commit')
                if not known:commits[request].append(r)
                a.update(r);consumed.add(receipt_key);row_inflight.pop((r['slot'],r['occupancy']))
                target_rows={active[x['slot']]['request'] for x in g['rows'] if x['slot'] in active}
                if g['active_rows']==2 and target_rows==set(requests) and all(any(c['op']==op for c in commits[key]) for key in requests):dual.append(op)
            elif kind in ('BOWN_STOP','BOWN_DONE'):
                request=r['request'];a=active.get(r['slot'])
                need(a and a['request']==request and a['occupancy']==r['occupancy'] and (r['slot'],r['occupancy']) not in row_inflight,'Stop/retirement addresses foreign occupancy')
                if kind=='BOWN_STOP':
                    need(request not in stops and dual and not any(key in done for key in requests),'Stop must follow real two-row overlap before either retirement')
                    need(r['last_op']==(commits[request][-1]['op'] if commits[request] else 0),'Stop ack op differs')
                    stops[request]=last_op
                else:
                    need(request not in done and r['watermark']==a['watermark'] and r['history']==a['history'],'Retirement differs from actual final cut')
                    need((request in stops and r['reason']=='cancel') or (request not in stops and r['reason'] in ('length','stop')),'Wrong native terminal reason')
                    done[request]=r;retired[request]=dict(a);active.pop(r['slot'])
            else:need(False,'Unknown native owned event')
        elif line.startswith('BT '):
            parts=line.split();need(len(parts)>=3,'Malformed native BT');slot=int(parts[1]);sample=int(parts[2]);a=active.get(slot)
            need(a and a['request'] not in stops and slot not in pending,'BT after stop or without native admission');pending[slot]=sample
    response_bt_counts={}
    for key,rows in commits.items():
        current=retired.get(key) or next((value for value in active.values() if value['request']==key),None)
        need(current is not None and current['watermark']-admissions[key]['watermark']==len(rows),'Response BT count differs from authenticated watermark delta')
        response_bt_counts[key]=current['watermark']-admissions[key]['watermark']
    if not sealed:return {'dual_operations':dual,'commits':commits,'stops':stops,'done':done,'admissions':admissions,'retired':retired,'histories':histories,'response_bt_counts':response_bt_counts,'prealigned':alignment_done,'pending_admissions':aligning,'aborted_admissions':aborted,'unconsumed_rows':row_inflight}
    need(dual and len(stops)==1 and len(done)==len(all_requests) and not active and not pending and not aligning and not row_inflight,'Missing overlap/cancellation/retirement evidence')
    canceled=next(iter(stops));survivor=next(x for x in requests if x!=canceled)
    need(any(c['op']>stops[canceled] for c in commits[survivor]),'Healthy occupancy never continued after peer stop ack')
    return {'schema':1,'scope':'native-two-slot-concurrent-decode-cancellation','requests':list(requests),'dual_operations':dual,'canceled':canceled,'survivor':survivor,'full_http_gate':False,'deployable':False}

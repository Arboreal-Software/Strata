#!/usr/bin/env python3
"""Validate captured native histories; never qualifies image/HTTP/reuse by itself."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

LIMIT=2 << 20
NEUTRAL={'temperature':0,'top_p':1,'top_k':1,'min_p':0,'penalty_last_n':0,
         'penalty_repeat':1,'penalty_freq':0,'penalty_present':0,'bias':False,'cvec_loaded':False}

def need(ok,message):
    if not ok:raise ValueError(message)

def strict(raw):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'Duplicate receipt key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite receipt')))

def history_digest(tokens):
    h=hashlib.sha256(b'strata-window-history-v1\0'+struct.pack('<Q',len(tokens)))
    for token in tokens:
        need(type(token) is int and 0<=token<=0x7fffffff,'Invalid token ID')
        h.update(struct.pack('<i',token))
    return h.hexdigest()

def validate(raw,expected_producer,requests,*,_sealed=True):
    need(isinstance(raw,bytes) and len(raw)<=LIMIT and raw.endswith(b'\n'),'Incomplete/oversized receipt file')
    need(isinstance(requests,list) and 1<=len(requests)<=128 and
         all(type(x.get('max_new')) is int and 1<=x['max_new']<=64 for x in requests), 'Bounded explicit expected requests required')
    lines=raw.splitlines();need(3<=len(lines)<=20000,'Receipt event bounds')
    records=[strict(x) for x in lines]
    producer=records[0]
    need(set(expected_producer)=={'pid','start_ticks','epoch','exe_sha256','argv'},'Explicit actual producer identity required')
    need(producer=={'kind':'producer','schema':1,**expected_producer},'Actual engine producer identity differs')
    need(type(producer.get('pid')) is int and producer['pid']>1 and
         type(producer.get('start_ticks')) is int and producer['start_ticks']>0 and
         re.fullmatch('[0-9a-f]{64}',producer.get('exe_sha256','')) and
         type(producer.get('epoch')) is int and producer['epoch']>0,'Invalid producer identity')
    if _sealed:
        need(records[-1]=={'kind':'eof','exit_code':0},'Actual native EOF/zero exit required')
        payload=records[1:-1]
    else:
        need(not any(x.get('kind')=='eof' for x in records),'Live prefix must be unsealed')
        payload=records[1:]
    rows=[];current=None
    for event in payload:
        need(set(event)=={'kind','line'} and event['kind']=='native' and isinstance(event['line'],str),'Unknown/forged event')
        line=event['line']
        if line.startswith('HTTP_ACCEPT '):
            need(current is None and len(rows)<len(requests),'Duplicate/hidden/native request')
            accepted=strict(line[len('HTTP_ACCEPT '):]);expected=requests[len(rows)]
            need(accepted.get('schema')==1 and accepted.get('request')==len(rows)+1 and
                 accepted.get('supported_serial_profile') is True,'Unsupported/order-mismatched native request')
            need(accepted.get('input_count')==len(expected['ids']) and
                 accepted.get('input_sha256')==history_digest(expected['ids']) and
                 accepted.get('max_new')==expected['max_new'],'Native accepted input/budget differs from independently encoded body')
            need(all(accepted.get(k)==v and (not isinstance(accepted.get(k),float) or math.isfinite(accepted[k]))
                     for k,v in NEUTRAL.items()),'Actual native sampling is not neutral')
            current={'accepted':accepted,'tokens':[]}
        elif line.startswith('T '):
            need(current is not None and re.fullmatch('T [0-9]+',line),'Token outside accepted native request')
            token=int(line[2:]);need(token<=0x7fffffff and len(current['tokens'])<current['accepted']['max_new'],'Output token/budget bound')
            current['tokens'].append(token)
        elif line.startswith('DONE '):
            need(current is not None,'Completion without native acceptance')
            fields=line.split()
            need(len(fields)>=6 and fields[1].isdigit() and fields[2].isdigit() and
                 int(fields[2])==current['accepted']['input_count'] and
                 int(fields[1])==len(current['tokens']) and current['tokens'] and
                 fields[5] in ('length','stop'),'Native completion/count/finish differs')
            current['finish']=fields[5];current['done']=line;rows.append(current);current=None
        else:raise ValueError('Native error/batched/unknown protocol cannot qualify serial receipt')
    need(current is None and len(rows)==len(requests),'Incomplete or missing native request')
    return {'kind':'native-http-history-binding-only','producer':producer,'requests':rows,
            'actual_input_digests_matched':True,'actual_emitted_token_ids':True,'sealed_native_exit':_sealed,
            'full_http_gate_passed':False,'reuse_proven':False,'two_slot_qualified':False,
            'qualified':False,'deployable':False}


def read_completed_prefix(raw,expected_producer,requests):
    """Unsealed observations only: build next conversation from genuine T IDs.
    Final admission MUST reread the real EOF/zero-exit artifact through validate().
    """
    result=validate(raw,expected_producer,requests,_sealed=False)
    result['kind']='unsealed-native-history-observation-only'
    return result

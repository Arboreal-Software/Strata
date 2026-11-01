"""Owned batch2 observations + native-issued stop nonce mirror, never cut authority."""
from dataclasses import dataclass
import hashlib
import json
import re
import struct
import threading
from pathlib import Path
from .native_receipts import NativeReceipt,identity

PATH=Path('/data/http-two-slot-receipts.jsonl')
PREFIXES=('HTTP_ACCEPT ','BOWN_','T ','DONE','BADM ','BT ','BDONE ','ERR')


def strict_json(raw):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate native observation field')
            out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite native observation')))


def request_key(value):return isinstance(value,str) and re.fullmatch('[0-9a-f]{12}',value) is not None

def history_digest(tokens):
    h=hashlib.sha256(b'strata-window-history-v1\0'+struct.pack('<Q',len(tokens)))
    for token in tokens:
        if type(token) is not int or not 0<=token<=0x7fffffff:raise ValueError('Invalid native request token')
        h.update(struct.pack('<i',token))
    return h.hexdigest()

@dataclass(frozen=True)
class StopNonce:
    request:str
    slot:int
    occupancy:int

class OwnedSlotObservations:
    """Only native ACCEPT may populate a stop nonce. Retirement clears live map.
    Two bounded historical records allow late scope cleanup to use its own old
    nonce; native rejection protects a newly reused slot, never a guessed stop.
    """
    def __init__(self):
        self.lock=threading.RLock();self.expected={};self.active={};self.retired={};self.failed=False
    def register(self,slot,request,tokens,budget):
        if type(slot) is not int or slot not in (0,1) or not request_key(request) or type(budget) is not int or not 1<=budget<=64:
            raise ValueError('Strict owned slot request binding invalid')
        expected={'request':request,'input_count':len(tokens),'input_sha256':history_digest(tokens),'full_budget':budget}
        with self.lock:
            if self.failed or slot in self.active:raise ValueError('Cannot replace live native occupancy')
            self.expected[slot]=expected
    def observe(self,line):
        if not line.startswith(('BOWN_ACCEPT ','BOWN_DONE ')):return
        try:
            prefix,raw=line.split(' ',1);row=strict_json(raw)
            slot=row['slot'];request=row['request'];occupancy=row['occupancy']
            if type(slot) is not int or slot not in (0,1) or not request_key(request) or type(occupancy) is not int or occupancy<=0:raise ValueError()
            token=StopNonce(request,slot,occupancy)
            with self.lock:
                if self.failed:raise ValueError()
                if prefix=='BOWN_ACCEPT':
                    expected=self.expected.get(slot)
                    if expected is None or slot in self.active or any(row.get(k)!=v for k,v in expected.items()):raise ValueError()
                    if any(old.slot==slot and old.occupancy==occupancy for old in (*self.active.values(),*self.retired.values())):raise ValueError()
                    self.active[slot]=token;self.expected.pop(slot,None)
                else:
                    if self.active.get(slot)!=token:raise ValueError()
                    self.active.pop(slot);self.retired[(slot,request)]=token
                    while len(self.retired)>2:self.retired.pop(next(iter(self.retired)))
        except Exception:
            with self.lock:self.failed=True
    def stop(self,slot,request):
        with self.lock:
            token=self.active.get(slot)
            if token is None or token.request!=request:token=self.retired.get((slot,request))
            if self.failed or token is None:raise ValueError('No native-issued matching occupancy for stop')
            return 'BSTOP %d occupancy=%d'%(token.slot,token.occupancy)

class TwoSlotReceipt(NativeReceipt):
    def observe(self,line):
        if line.startswith(PREFIXES):self.write({'kind':'native','line':line.rstrip('\n')})
    @classmethod
    def enabled(cls,pid,epoch,env=None):
        if (env or {}).get('STRATA_HTTP_TWO_SLOT_RECEIPTS')!='1':return None
        try:return cls(PATH,identity(pid,epoch))
        except Exception:return None

class OwnedTerminalPairing:
    """Join native terminal and authenticated retirement in either arrival order.

    Native cancellation retires before BDONE; ordinary completion sends BDONE
    before its quiet retirement. Neither half alone releases a response lane.
    Each pending half is pinned to the actual accepted occupancy, not just slot.
    """
    def __init__(self, observations):
        self.observations=observations
        self.terminals={}
        self.retirements={}
        self.cancel_terminals={}

    @staticmethod
    def _slot(value):
        if type(value) is not int or value not in (0,1):
            raise ValueError('Invalid owned terminal slot')
        return value

    def observe(self, line):
        try:
            return self._observe(line)
        except (ValueError,KeyError,IndexError,TypeError):
            with self.observations.lock:self.observations.failed=True
            raise ValueError('Conflicting or unbound owned terminal/retirement') from None

    def _observe(self, line):
        observations=self.observations
        if observations.failed:raise ValueError('Owned observation epoch failed')
        slot=None
        if line.startswith('BDONE '):
            fields=line.split()
            if len(fields)!=5 or not fields[1].isdigit() or not fields[2].isdigit():
                raise ValueError('Malformed owned terminal')
            slot=self._slot(int(fields[1]))
            if fields[3] not in ('cancel','length','stop','nonfinite','shutdown'):
                raise ValueError('Invalid owned terminal')
            previous=self.terminals.get(slot)
            if previous is not None:
                # fa may receive BSTOP while a completed lane awaits quiet
                # retirement: its real length/stop terminal is followed by a
                # real cancel terminal. Hold this second half provisionally;
                # ONLY matching authenticated BOWN_DONE(cancel) can select it.
                if previous[2] not in ('length','stop') or fields[3]!='cancel' or \
                        int(fields[2])!=previous[3] or slot in self.cancel_terminals:
                    raise ValueError('Duplicate/conflicting owned terminal')
                self.cancel_terminals[slot]=(previous[0],line,fields[3],int(fields[2]))
            with observations.lock:
                owner=observations.active.get(slot)
            if owner is None:
                retired=self.retirements.get(slot)
                owner=retired[0] if retired else None
            if owner is None:raise ValueError('Owned terminal lacks actual occupancy')
            if previous is None:self.terminals[slot]=(owner,line,fields[3],int(fields[2]))
        elif line.startswith(('BOWN_ACCEPT ','BOWN_DONE ')):
            kind,raw=line.split(' ',1);row=strict_json(raw)
            slot=self._slot(row['slot'])
            if kind=='BOWN_ACCEPT':
                if slot in self.terminals or slot in self.retirements:
                    raise ValueError('New occupancy before prior terminal/retirement pair')
            else:
                if slot in self.retirements:raise ValueError('Duplicate owned retirement')
                with observations.lock:owner=observations.active.get(slot)
                if owner is None:raise ValueError('Retirement lacks live native occupancy')
                self.retirements[slot]=(owner,row)
            observations.observe(line)
            if observations.failed:raise ValueError('Malformed/foreign owned observation')
        else:
            # A terminal/retirement half closes this lane to further native output,
            # while the independent peer remains routable.
            if line.startswith(('BT ','BOWN_COMMIT ','BOWN_KNOWN ')):
                if line.startswith('BT '):slot=self._slot(int(line.split()[1]))
                else:slot=self._slot(strict_json(line.split(' ',1)[1])['slot'])
                if slot in self.terminals or slot in self.retirements:
                    raise ValueError('Native work after owned terminal/retirement')
            return None
        terminal=self.terminals.get(slot);retired=self.retirements.get(slot)
        if terminal is None or retired is None:return None
        if terminal[0]!=retired[0]:
            raise ValueError('Owned terminal/retirement identity differs')
        replacement=self.cancel_terminals.get(slot)
        if retired[1]['reason']=='cancel' and terminal[2] in ('length','stop'):
            if replacement is None:return None  # actual cancel terminal still required
            if replacement[0]!=retired[0] or replacement[3]!=terminal[3]:
                raise ValueError('Cancellation replacement differs from actual occupancy/count')
            terminal=replacement
        elif terminal[2]!=retired[1]['reason'] or replacement is not None:
            raise ValueError('Owned terminal/retirement reason differs')
        self.terminals.pop(slot);self.retirements.pop(slot);self.cancel_terminals.pop(slot,None)
        return slot,terminal[1]  # original native line, never a synthetic completion

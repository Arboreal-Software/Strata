"""Bounded opt-in observation of ACTUAL native stdout, never planned inputs."""
import hashlib
import json
import os
from pathlib import Path

PREFIX='HTTP_ACCEPT '
MAX_BYTES=2 << 20
MAX_EVENTS=20000
PATH=Path('/data/http-native-receipts.jsonl')


def identity(pid,epoch):
    proc=Path('/proc')/str(pid)
    raw=(proc/'stat').read_text()
    sha=hashlib.sha256()
    with (proc/'exe').open('rb') as stream:
        for chunk in iter(lambda:stream.read(1 << 20),b''):sha.update(chunk)
    return {'kind':'producer','schema':1,'pid':pid,'start_ticks':int(raw.rsplit(')',1)[1].split()[19]),
            'epoch':epoch,'exe_sha256':sha.hexdigest(),
            'argv':(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')}


class NativeReceipt:
    def __init__(self,path,producer):
        self.path=Path(path);self.bytes=0;self.events=0;self.failed=False;self.stream=None
        try:
            # Fixed private tmpfs path in production. No append/reuse/symlink-follow.
            fd=os.open(self.path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
            self.stream=os.fdopen(fd,'wb',buffering=0)
            self.write(producer)
        except Exception:self.failed=True;self.close()
    def write(self,row):
        if self.failed or self.stream is None:return
        try:
            raw=(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode()
            if self.bytes+len(raw)>MAX_BYTES or self.events>=MAX_EVENTS:raise ValueError('Native receipt output cap')
            self.stream.write(raw);self.bytes+=len(raw);self.events+=1
        except Exception:
            self.failed=True
            # Incomplete/truncated artifacts cannot satisfy the strict reader.
            self.close()
    def observe(self,line):
        if line.startswith((PREFIX,'T ','DONE','ERR','BT ','BDONE ')):
            self.write({'kind':'native','line':line.rstrip('\n')})
    def close(self):
        if self.stream is not None:
            try:self.stream.close()
            except Exception:pass
            self.stream=None
    @classmethod
    def enabled(cls,pid,epoch,env=None):
        if (os.environ if env is None else env).get('STRATA_HTTP_NATIVE_RECEIPTS')!='1':return None
        try:return cls(PATH,identity(pid,epoch))
        except Exception:return None

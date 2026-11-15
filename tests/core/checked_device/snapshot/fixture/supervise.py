# Prepared only. Streamed bounded output, exact process ownership and diagnostics.
import os,selectors,subprocess,sys,time
cases={
 'get':('FAULT get 1','OnDevice get device'),
 'select':('FAULT set 1','OnDevice select device'),
 'restore':('FAULT set 2','OnDevice restore device'),
 'nested-restore':('FAULT set 3','OnDevice restore device'),
 'parent-restore':('FAULT set 4','OnDevice restore device'),
 'worker':('FAULT set 1','OnDevice select device'),
 'invalid':('', 'OnDevice invalid device'),
 'partial-no-device':('', 'OnDevice invalid device'),
 'no-work':None,'success':None,'legacy':None}
def captured(command):
 p=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
 buffers=[bytearray(),bytearray()];sel=selectors.DefaultSelector()
 try:
  for i,pipe in enumerate((p.stdout,p.stderr)):
   os.set_blocking(pipe.fileno(),False);sel.register(pipe,selectors.EVENT_READ,i)
  end=time.monotonic()+5
  while sel.get_map():
   remaining=end-time.monotonic()
   if remaining<=0:raise TimeoutError('child output deadline')
   for key,_ in sel.select(min(remaining,.05)):
    chunk=os.read(key.fileobj.fileno(),min(4096,8193-sum(map(len,buffers))))
    if not chunk:sel.unregister(key.fileobj);continue
    buffers[key.data].extend(chunk)
    if sum(map(len,buffers))>8192:raise RuntimeError('child output cap')
  p.wait(timeout=max(.001,end-time.monotonic()))
  return p.returncode,bytes(buffers[0]),bytes(buffers[1])
 finally:
  sel.close()
  # poll/reap before signal avoids signalling an already-reaped PID. This
  # supervisor is the sole reaper and fixture binaries spawn no descendants.
  if p.poll() is None:p.kill()
  p.wait(timeout=2)
  p.stdout.close();p.stderr.close()
for name,want in cases.items():
 code,out,err=captured([sys.argv[1],name]);assert out==b'',(name,out)
 if want:
  expected=('AFTER CHILD\n' if name=='parent-restore' else '')
  if want[0]:expected+=want[0]+'\n'
  expected+='strata: strict retirement failed at '+want[1]+'; worker restart required\n'
  assert code==1 and err==expected.encode(),(name,code,err)
 else:
  assert code==0 and err==(b'AFTER CHILD\n' if name=='success' else (b'FAULT get 1\nFAULT set 1\nFAULT set 3\n' if name=='legacy' else b'')),(name,code,err)

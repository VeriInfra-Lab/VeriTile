"""Replay actual installed JIT binder, on real CPU tensor metadata only."""
from pathlib import Path
import hashlib,importlib.util,inspect,json,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;O=W/'binder-001';O.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
import torch,triton
import triton.runtime.jit as jit
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
source=W/'replay/evidence/source-prefix.py'
spec=importlib.util.spec_from_file_location('independent_binder_source',source)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
backend=CUDABackend(GPUTarget('cuda',90,32))
binder=jit.create_function_from_signature(mod.add_kernel.signature,mod.add_kernel.params,backend)
native=W/'replay'/load(W/'replay/evidence/native-latest.json')['receipt']
frontend=load(native.with_name('records.json'))
names=list(dict.fromkeys(r['x'] for r in frontend));tensors={n:torch.empty(0,dtype=getattr(torch,n)) for n in names}
records=[]
for f in frontend:
 xn,yn=f['x'],f['y'];row={'x':xn,'y':yn}
 try:
  bound,sig,opts=binder(tensors[xn],tensors[yn],tensors[xn],1,4)
  row.update(status='bound',signature=sig,options=opts)
  assert bound['n_elements']==1 and bound['BLOCK_SIZE']==4
  assert sig[3]==('i32',1) and sig[4]==('constexpr',4)
  actual={k:sig[i][0] for i,k in enumerate(['in_ptr0','in_ptr1','out_ptr','n_elements'])}
  assert actual==f['signature'],(row,f)
 except KeyError as e:
  assert f.get('error_type')=='KeyError',f
  row.update(status='binder_rejected',error_type=type(e).__name__,error=str(e))
 records.append(row)
assert len(records)==484
assert sum(r['status']=='binder_rejected' for r in records)==195
(O/'records.json').write_text(json.dumps(records,indent=2)+'\n')
(O/'create_function_from_signature.py.txt').write_text(inspect.getsource(jit.create_function_from_signature))
result={'status':'passed','records':len(records),'bound':289,'binder_rejected':195,'n1_stays_i32':True,'source_sha256':sha(source),'fresh_frontend_records_sha256':sha(native.with_name('records.json')),'installed_jit_sha256':sha(jit.__file__),'torch':torch.__version__,'triton':triton.__version__,'GPU_launches':0,'scope':'Actual installed JIT binder using real empty CPU tensor dtype metadata; no device/context query or GPU launch','artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

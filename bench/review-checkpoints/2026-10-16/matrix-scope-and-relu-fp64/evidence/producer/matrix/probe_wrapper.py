"""Replay exact wrapper AST with trusted Torch metadata and explicit call failures.
CUDA requests are recorded then redirected to meta tensors; there is no GPU run.
"""
import ast,hashlib,itertools,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import torch
ROOT=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
OUT=Path(__file__).resolve().parent/'evidence'
SOURCE=ROOT/'bench/tritonbench_g/matrix_transpose/matrix_transpose.py'
node=next(n for n in ast.parse(SOURCE.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='wrapper')
module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
class Injected(Exception):pass

def arange(n):
 if n>=2**32:return 'indexWidth'
 if n==0:return 'nonPositive'
 if n&(n-1):return 'nonPowerOfTwo'
 if n>2**20:return 'tensorSize'
 return None

def admit(m,d):
 if (e:=arange(m)) is not None:return 'sizeMRange.'+e
 if (e:=arange(d)) is not None:return 'dHeadRange.'+e
 if m*d>2**20:return 'matrixPointerTensorSize'
 return None

def replay(m,d,failure=()):
 trace=[];tensors=[];launched=[]
 class TorchMeta:
  float16=torch.float16
  def call(self,name,shape,**kw):
   trace.append(name)
   assert kw=={'dtype':torch.float16,'device':'cuda'}
   if name in failure:raise Injected(name)
   t=getattr(torch,name)(shape,dtype=kw['dtype'],device='meta');tensors.append(t);return t
  def randn(self,shape,**kw):return self.call('randn',shape,**kw)
  def zeros(self,shape,**kw):return self.call('zeros',shape,**kw)
 class Kernel:
  def __getitem__(self,grid):
   assert grid==(1,)
   def launch(M,Out,sx,sy,ox,oy,sm,dh):
    trace.append('launch')
    if 'launch' in failure:raise Injected('launch')
    assert (sm,dh)==(m,d)
    assert M is tensors[0] and Out is tensors[1]
    assert M.dtype==Out.dtype==torch.float16
    assert tuple(M.shape)==(m,d) and tuple(Out.shape)==(d,m)
    assert (sx,sy)==M.stride()==(max(d,1),1)
    assert (ox,oy)==Out.stride()==(max(m,1),1)
    assert M.is_contiguous() and Out.is_contiguous()
    if (e:=admit(m,d)) is not None:raise Injected('frontend.'+e)
    if 'backend' in failure:raise Injected('backend')
    launched.append(Out)
   return launch
 env={'torch':TorchMeta(),'kernel':Kernel()}
 exec(compile(module,str(SOURCE),'exec'),env)
 row={'m':m,'d':d,'failure_flags':list(failure)}
 try:out=env['wrapper'](m,d)
 except Exception as exc:row.update(status='error',error_kind=str(exc) if isinstance(exc,Injected) else 'torch.'+type(exc).__name__,error_type=type(exc).__name__)
 else:
  assert launched==[out] and out is tensors[1]
  trace.append('returned')
  row.update(status='returned',output_shape=list(out.shape),output_stride=list(out.stride()),output_dtype=str(out.dtype),return_identity=True)
 row['trace']=trace
 return row

success=[replay(2**a,2**b) for a in range(21) for b in range(21-a)]
assert len(success)==231 and all(r['status']=='returned' and r['trace']==['randn','zeros','launch','returned'] for r in success)
failures=[]
for m,d in [(16,16),(32,64),(0,16),(16,3),(2**20,2),(2**21,0)]:
 for bs in itertools.product([False,True],repeat=4):
  flags=[n for n,b in zip(['randn','zeros','launch','backend'],bs) if b]
  r=replay(m,d,flags)
  early=next((f for f in ['randn','zeros','launch'] if f in flags),None)
  frontend=admit(m,d)
  expected=early or ('frontend.'+frontend if frontend else None) or ('backend' if 'backend' in flags else None)
  assert r['status']==('error' if expected else 'returned')
  if expected:assert r['error_kind']==expected
  failures.append(r)
invalid=[]
for m,d in [(-1,4),(4,-1),(1.5,4),(4,1.5),(True,4),(4,False),(None,4),('4',4)]:
 r=replay(m,d);assert r['status']=='error' and 'launch' not in r['trace'];invalid.append(r)
result={'status':'passed','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'wrapper_ast_sha256':hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),'torch_version':torch.__version__,'admitted_metadata_cases':success,'source_failure_cases':failures,'invalid_shape_boundary_cases':invalid,'trusted_contracts':['Torch randn/zeros return fresh contiguous fp16 tensors of requested shape on CUDA when their primitive call succeeds.','randn samples are arbitrary fp16 words; no probability distribution is used.','Torch allocation failure, Python shape conversion and unsupported caller types are ordinary allocation-call primitive failures (the observed boolean second argument reaches zeros before rejection).','Triton JIT/driver and backend/runtime errors propagate from the single launch call. Successful execution is proved by the actual interpreter; there is no arbitrary final-value result.'],'limitations':['CUDA requests redirected to meta device for metadata evidence.','Injected errors test source call wiring only; they do not prove allocator/runtime implementation or all Python caller classifications.','No GPU or full native backend execution.']}
(OUT/'wrapper-binding.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','metadata_cases':len(success),'failure_flag_cases':len(failures),'invalid_shape_cases':len(invalid)}))

"""Source/binder audit; metadata-only wrapper harness, no GPU numerical claim."""
import ast, copy, hashlib, importlib.util, json, math, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'.veritile-cache/kernel-evidence/layer_norm_fwd_nonzero_tile'
ORIGINAL=ROOT/'bench/repairs/tritonbench_g/layer_norm_fwd/layer_norm_fwd_repaired.py'
SOURCE=HERE/'layer_norm_fwd_nonzero_tile.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n):return ast.dump(n,include_attributes=False)
old=ast.parse(ORIGINAL.read_text());new=ast.parse(SOURCE.read_text())
expected=copy.deepcopy(old);changes=0
for node in ast.walk(expected):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BLOCK_N' for t in node.targets):
  node.value=ast.Call(ast.Name('max',ast.Load()),[ast.Constant(1),node.value],[]);changes+=1
assert changes==1 and dump(expected)==dump(new)
kernel=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='_layer_norm_fwd_1pass_kernel')
old_kernel=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==kernel.name)
assert dump(kernel)==dump(old_kernel)
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.compiler import make_backend
from triton.backends.compiler import GPUTarget
k=copy.deepcopy(kernel);k.decorator_list=[ast.Attribute(ast.Name('triton',ast.Load()),'jit',ast.Load())]
extracted=OUT/'candidate_selected_jit.py'
extracted.write_text('import triton\nimport triton.language as tl\n'+ast.unparse(k)+'\n')
spec=importlib.util.spec_from_file_location('candidate_layer_norm_jit',extracted)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
fn=mod._layer_norm_fwd_1pass_kernel
binder=create_function_from_signature(fn.signature,fn.params,make_backend(GPUTarget('cuda',80,32)))
allocations=[]
class Tensor:
 def __init__(self,shape,dtype=torch.float32,device=None):
  self.shape=tuple(shape);self.dtype=dtype;self.device=device or SimpleNamespace(type='cuda',index=0)
  self.layout=torch.strided;self.ndim=len(shape);self.ptr=4096+256*len(allocations);allocations.append(self)
 def stride(self,axis=None):
  stride=(max(self.shape[1],1),1) if self.ndim==2 else (1,)
  return stride if axis is None else stride[axis]
 def storage_offset(self):return 0
 def is_neg(self):return False
 def numel(self):return math.prod(self.shape)
 def data_ptr(self):return self.ptr
 def element_size(self):return torch.empty((),dtype=self.dtype).element_size()
 def is_contiguous(self):return True
 def tensor_split(self,count,dim):
  assert count==2 and dim==0
  return tuple(Tensor((self.shape[0]//2,*self.shape[1:]),self.dtype,self.device) for _ in range(2))
class DeviceContext:
 def __enter__(self):return self
 def __exit__(self,*args):pass
class Launcher:
 def __init__(self):self.calls=[]
 def __getitem__(self,grid):
  def launch(*args):
   kw=dict(HAS_X1=args[5] is not None,HAS_W1=args[6] is not None,HAS_B1=args[7] is not None)
   params,specializations,options=binder(*args,**kw)
   self.calls.append(dict(grid=grid,args=args,params=params,specializations=specializations,options=options))
  return launch
launcher=Launcher()
def empty(*shape,dtype,device):
 if len(shape)==1 and isinstance(shape[0],tuple):shape=shape[0]
 return Tensor(shape,dtype,device)
namespace={'math':math,'triton':triton,'_layer_norm_fwd_1pass_kernel':launcher,
 'torch':SimpleNamespace(Tensor=Tensor,dtype=torch.dtype,float32=torch.float32,int64=torch.int64,bool=torch.bool,
  strided=torch.strided,empty=empty,empty_like=lambda x,dtype=None:Tensor(x.shape,x.dtype if dtype is None else dtype,x.device),
  randint=lambda high,shape,device,dtype:Tensor(shape,dtype,device),cuda=SimpleNamespace(device=lambda _:DeviceContext()))}
wrapper=ast.Module(body=[n for n in new.body if isinstance(n,ast.FunctionDef) and n.name!='_layer_norm_fwd_1pass_kernel'],type_ignores=[])
exec(compile(wrapper,str(SOURCE),'exec'),namespace)
records=[]
for M,N,options in [(2,0,{}),(0,0,{}),(0,17,{}),(2,1,{}),(2,16384,{}),
 (2,0,dict(is_rms_norm=True)),(2,0,dict(dropout_p=0.25,return_dropout_mask=True)),
 (2,0,dict(eps=5e-324,out_dtype=torch.float64,residual_dtype=torch.int64)),
 (2,0,dict(eps=1e308,dropout_p=1-2**-53))]:
 x=Tensor((M,N));w=Tensor((N,),device=x.device)
 result=namespace['_layer_norm_fwd'](x,w,None,options.pop('eps',1.0),**options)
 call=launcher.calls[-1];params=call['params']
 assert params['BLOCK_N']==max(1,min(16384,triton.next_power_of_2(N)))
 assert params['M']==M and params['N']==N and call['grid']==(M,)
 assert result[2] is None if options.get('is_rms_norm',False) else result[2].shape==(M,)
 assert result[3].shape==(M,) and result[0].shape==(M,N)
 records.append(dict(M=M,N=N,tile=params['BLOCK_N'],grid=list(call['grid']),
  out_dtype=str(result[0].dtype),residual_dtype=str(result[4].dtype),
  mean_present=result[2] is not None,returned_shapes=[None if t is None else list(t.shape) for t in result],
  scalar_words={'eps':__import__('struct').pack('>d',params['eps']).hex(),'p':__import__('struct').pack('>d',params['dropout_p']).hex()},
  specialization={name:list(value) for name,value in zip(fn.arg_names,call['specializations']) if name in ('M','N','eps','dropout_p','BLOCK_N')}))
# The width failure remains after allocation and before launch.
x=Tensor((2,16385));w=Tensor((16385,),device=x.device);before=len(allocations);calls=len(launcher.calls)
try:namespace['_layer_norm_fwd'](x,w,None,1.0)
except RuntimeError:pass
else:raise AssertionError('overwidth accepted')
assert len(allocations)>before and len(launcher.calls)==calls
# Reuse the frozen compiler control: identical JIT body and exact fixed tile.
prior=ROOT/'.veritile-cache/layer-norm-fwd-zero-width-diagnostic/report.json'
report=json.loads(prior.read_text());control=next(r for r in report['records'] if r['block']==1)
assert control['status']=='compiled'
body=dump(ast.Module(body=kernel.body,type_ignores=[]))
assert hashlib.sha256(body.encode()).hexdigest()==report['kernel_body_ast_sha256']
for path,pin in control['artifacts'].items():assert sha(Path(path))==pin
result={'status':'PASS','source_sha256':sha(SOURCE),'original_sha256':sha(ORIGINAL),
 'sole_ast_change':'BLOCK_N = max(1, original expression)','kernel_decorators_and_body_unchanged':True,
 'records':records,'overwidth_after_allocations':True,'triton':triton.__version__,
 'installed_binder':'create_function_from_signature, CUDA SM80 backend',
 'reused_fixed_tile_compiler_control':{'report':str(prior),'sha256':sha(prior),'record':control},
 'scope':'Metadata-only exact guard/wrapper execution and installed binder. No GPU launch/numerics. Frozen original failures are not rerun; unchanged-body BLOCK_N=1 offline compiler control is hash-verified and reused.'}
(OUT/'candidate-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'PASS','wrapper_cases':len(records),'sole_ast_change':True,'fixed_tile_compiler_control_reused':True}))

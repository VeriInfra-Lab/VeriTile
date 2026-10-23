"""Candidate branch-local scalar assertion, raw identity, and actual JIT ABI."""
from pathlib import Path
from types import SimpleNamespace
import argparse,contextlib,hashlib,importlib.util,inspect,json,os,sys
sys.dont_write_bytecode=True
os.environ.pop('TRITON_INTERPRET',None)
ROOT=Path(__file__).resolve().parents[4]
SOURCE=Path(__file__).with_name('fused_rwkv6_kernel_scalar_abi_checked.py')
ORIGINAL=ROOT/'bench/repairs/tritonbench_g/fused_rwkv6_kernel/fused_rwkv6_kernel_repaired.py'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'.veritile-cache/kernel-evidence/fused_rwkv6_kernel_scalar_abi_checked/scalar-abi')
OUT=parser.parse_args().output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(OUT/'cache')
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton.compiler import ASTSource
def pin(p):
 return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert pin(ORIGINAL)['sha256']=='dfd768e5f6f3c8b439ef7e76b26a1330e2e81c9071c2d4c091fb8b4122221669'
import ast
old=ast.parse(ORIGINAL.read_text());new=ast.parse(SOURCE.read_text())
old_kernel=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='fused_recurrent_rwkv6_fwd_kernel')
new_kernel=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='fused_recurrent_rwkv6_fwd_kernel')
assert ast.dump(old_kernel)==ast.dump(new_kernel)
# Removing the sole new branch assertion restores the exact original AST.
new_forward=next(n for n in new.body if isinstance(n,ast.ClassDef)).body[0]
launch=next(n for n in new_forward.body if isinstance(n,ast.If) and ast.unparse(n.test)=='T or output_final_state')
assert ast.dump(launch.body[0].test)==ast.dump(ast.parse('type(scale) is int and not -(2**63) <= scale < 2**64', mode='eval').body)
del launch.body[0]
assert ast.dump(old)==ast.dump(new)
spec=importlib.util.spec_from_file_location('rwkv_scale_probe',SOURCE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
kernel=m.fused_recurrent_rwkv6_fwd_kernel
class Tensor:
 def __init__(self,data):
  self.data=data;self.device=torch.device('cuda:0');self.shape=data.shape
  self.ndim=data.ndim;self.dtype=data.dtype;self.layout=data.layout
 def numel(self):return self.data.numel()
 def is_contiguous(self):return self.data.is_contiguous()
 def is_neg(self):return self.data.is_neg()
 def is_conj(self):return self.data.is_conj()
 def data_ptr(self):return self.data.data_ptr()
 def new_empty(self,*shape,dtype=None):return Tensor(torch.empty(shape,dtype=self.dtype if dtype is None else dtype))
 def sum(self,dim):return Tensor(self.data.sum(dim))
 def to(self,dtype):return Tensor(self.data.to(dtype))
class Context:
 def save_for_backward(self,*args):self.saved=args
calls={}
class Capture:
 def __getitem__(self,grid):
  calls['grid']=grid
  def launch(*args,**kwargs):calls.update(args=args,kwargs=kwargs)
  return launch
m.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
    cuda=SimpleNamespace(device=lambda device:contextlib.nullcontext()))
m.fused_recurrent_rwkv6_fwd_kernel=Capture()
target=GPUTarget('cuda',80,32);backend=CUDABackend(target)
binder=create_function_from_signature(kernel.signature,kernel.params,backend)
profiles=[]
cases=[(label,scale,1,False) for label,scale in [('fp32_control',0.5),('i32_control',2),('i64_control',2**31),
 ('i64_min',-(2**63)),('u64_control',2**63),('u64_max',2**64-1),('above_u64',2**64),
 ('large_integer',2**100),('below_i64',-(2**63)-1),('float_inf',float('inf')),('float_neg_inf',float('-inf')),('float_nan',float('nan')),('large_float',1e300),('minus_zero',-0.0)]]
cases += [('zero_time_no_final',2**100,0,False),('zero_time_no_final_negative',-(2**100),0,False),('zero_time_final',2**64,0,True),('zero_time_final_legal',2**64-1,0,True)]
for label,scale,T,wants_final in cases:
 tensors=[Tensor(torch.ones(1,1,T,1)) for _ in range(4)]+[Tensor(torch.ones(1,1))]
 calls.clear();ctx=Context()
 rejected=None
 try:
  out,final=m.FusedRecurrentRWKV6Function.forward(ctx,*tensors,scale,None,wants_final)
 except Exception as error:
  rejected=error
 if rejected is None:
  assert ctx.scale is scale and list(out.shape)==[1,1,T,1] and (final is not None)==wants_final
 record={'case':label,'scale':str(scale),'python_type':type(scale).__name__,
   'T':T,'output_final_state':wants_final,'wrapper_admitted':rejected is None,'source':pin(SOURCE)}
 if rejected is not None:
  assert type(rejected) is ValueError and str(rejected)=='launched integer scale must fit the Triton i64/u64 ABI'
  assert not calls and (T or wants_final) and type(scale) is int and not -(2**63)<=scale<2**64
  record.update(status='assertion_rejected',exception=type(rejected).__name__,diagnostic=str(rejected))
  profiles.append(record);print(label,record['status'],flush=True);continue
 if not calls:
  assert T==0 and not wants_final
  record.update(status='no_launch',scale_saved_unchanged=True)
  profiles.append(record);print(label,'no_launch',flush=True);continue
 assert calls['grid']==(1,1,1) and calls['args'][10] is scale
 record['actual_launch_scale_preserved']=True
 args=[x.data if isinstance(x,Tensor) else x for x in calls['args']]
 try:
  bound,specialization,options=binder(*args,**calls['kwargs'])
  assert list(bound)==kernel.arg_names and bound['scale'] is scale
  _,signature,constants,attrs=kernel._pack_args(backend,options,bound,specialization,None)
  record.update(status='bound',signature=signature,constants={kernel.arg_names[k[0]]:v for k,v in constants.items()},options=options)
 except Exception as error:
  record.update(status='binding_rejected',exception=type(error).__name__,diagnostic=str(error))
 else:
  if label in ('fp32_control','u64_max','zero_time_final_legal'):
   code=triton.compile(ASTSource(kernel,signature,constexprs=constants,attrs=attrs),target=target,options=options)
   artifacts=[]
   for kind in ('ttir','ttgir','llir','ptx'):
    p=OUT/(label+'.'+kind);p.write_text(code.asm[kind]);artifacts.append(pin(p))
   record.update(status='compiled',artifacts=artifacts,metadata=code.metadata._asdict())
 profiles.append(record);print(label,record['status'],record.get('diagnostic',''),flush=True)
assert len(profiles)==18
assert [x['case'] for x in profiles if x['status']=='assertion_rejected']==['above_u64','large_integer','below_i64','zero_time_final']
assert not any(x['status']=='binding_rejected' for x in profiles)
import triton.runtime.jit as jit
import triton._C.libtriton as native
report={'task':'fused_rwkv6_kernel_scalar_abi_checked','status':'PASS',
 'binder_source':pin(Path(inspect.getfile(jit.create_function_from_signature))),
 'native_specializer_binary':pin(Path(native.__file__)),'source':pin(SOURCE),'original':pin(ORIGINAL),'sole_change':'branch-local raw-int ABI guard; kernel AST and all other source AST unchanged','checker':pin(Path(__file__)),
 'triton_version':triton.__version__,'GPU_execution':False,'profiles':profiles,
 'scope':'Actual candidate checked forward wrapper on CPU-backed Tensor metadata with CUDA identity/context adaptation; exact captured args use real JIT binder/packer. Small live B=H=T=K=V=1.'}
p=OUT/'report.json';p.write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(pin(p)))

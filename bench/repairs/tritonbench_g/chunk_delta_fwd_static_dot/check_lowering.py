"""Actual wrapper/JIT binding controls for static dot-tile admission; no GPU."""
from pathlib import Path
from types import SimpleNamespace
import re,argparse,contextlib,hashlib,importlib.util,inspect,json,os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'bench/repairs/tritonbench_g/chunk_delta_fwd_static_dot/chunk_delta_fwd_static_dot.py'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'.veritile-cache/kernel-evidence/chunk_delta_fwd_static_dot/lowering')
OUT=parser.parse_args().output.resolve()
OUT.mkdir(parents=True,exist_ok=True)
os.environ['TRITON_CACHE_DIR']=str(OUT/'compiler-cache')
os.environ.pop('TRITON_INTERPRET',None)
import torch,triton
from triton.runtime.jit import create_function_from_signature
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend,min_dot_size
from triton.compiler import ASTSource
import triton.language.semantic

def pin(p):
 return {'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
assert pin(SOURCE)['sha256']=='591848f81de55559a1c48131f7bbe30bb865f742d342202862ff9fcf6aa7f523'
spec=importlib.util.spec_from_file_location('delta_min_dot',SOURCE)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
autotuner=m.chunk_delta_rule_fwd_kernel_h;kernel=autotuner.fn
configs=[config.num_warps for config in autotuner.configs]
assert configs==[1,2,4,8,16,32]
class Tensor:
    def __init__(self, data):
        self.data = data
        self.device = torch.device("cuda:0")
        self.is_cuda = True

    @property
    def shape(self): return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    @property
    def ndim(self): return self.data.ndim
    @property
    def layout(self): return self.data.layout
    def stride(self, i=None): return self.data.stride() if i is None else self.data.stride(i)
    def numel(self): return self.data.numel()
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def is_conj(self): return self.data.is_conj()
    def data_ptr(self): return self.data.data_ptr()
    def untyped_storage(self): return self.data.untyped_storage()
    def element_size(self): return self.data.element_size()



calls={}
class Capture:
 def __getitem__(self,grid):
  calls['grid']=grid
  def launch(*args,**kwargs):calls.update(args=args,kwargs=kwargs)
  return launch
m.torch=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
 empty=lambda shape,dtype,device:Tensor(torch.empty(shape,dtype=dtype)),
 cuda=SimpleNamespace(device=lambda device:contextlib.nullcontext()))
m.chunk_delta_rule_fwd_kernel_h=Capture()
target=GPUTarget('cuda',80,32);backend=CUDABackend(target)
binder=create_function_from_signature(kernel.signature,kernel.params,backend)

profiles=[]
cases=[('zero_time',0,9,3,16,False,False),('zero_time_states',0,9,3,16,True,True),
       ('single_none',17,9,3,16,False,False),('single_seed',17,9,3,16,True,False),
       ('single_final',17,9,3,16,False,True),('multiple_both',85,9,3,80,True,True),
       ('rounded_overlap',33,129,3,17,True,True)]
for label,T,K,V,BT,want_seed,want_final in cases:
 inputs=[Tensor(torch.ones((1,1,T,K),dtype=torch.float32)) for _ in range(2)]
 inputs.append(Tensor(torch.ones((1,1,T,V),dtype=torch.float32)))
 seed=Tensor(torch.ones((1,1,K,V),dtype=torch.float32)) if want_seed else None
 final=Tensor(torch.empty((1,1,K,V),dtype=torch.float32)) if want_final else None
 calls.clear();returned=m.chunk_fwd_h_fn_static_dot(*inputs,BT,seed,final)
 assert calls['grid']==(1,1,1)
 args=[x.data if isinstance(x,Tensor) else x for x in calls['args']]
 bound,specialization,options=binder(*args,**calls['kwargs'],num_warps=configs[0])
 assert list(bound)==kernel.arg_names
 for name,value in zip(kernel.arg_names[:7],args[:7]):assert bound[name] is value
 expected=dict(zip(kernel.arg_names[7:len(args)],args[7:]));expected.update(calls['kwargs'])
 for name,value in expected.items():assert bound[name]==value
 _,signature,constants,attrs=kernel._pack_args(backend,options,bound,specialization,None)
 named={kernel.arg_names[key[0]]:value for key,value in constants.items()}
 for name,value in expected.items():
  if signature[name]=='constexpr':assert named[name]==value
 assert named['BK']>=16 and named['BC']>=16
 compiled=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants,attrs=attrs),
     target=target,options=options)
 artifacts=[]
 for ext in ('ttir','ttgir','llir','ptx'):
  path=OUT/(label+'.'+ext);path.write_text(compiled.asm[ext]);artifacts.append(pin(path))
 ir=compiled.asm['ttir'];ptx=compiled.asm['ptx']
 assert 'mma.sync' not in ptx and compiled.metadata.enable_fp_fusion is True
 if T:
  assert 'fma.rn.f32' in ptx
  dots=re.findall(r'(%\w+) = tt.dot [^,\n]+, [^,\n]+, (%\w+) ',ir)
  assert len(dots)==2,dots
  assert re.search(re.escape(dots[0][1])+r' = arith.constant dense<0.000000e\+00>',ir)
  assert re.search(r'arith.subf .*'+re.escape(dots[0][0]),ir)
  if triton.cdiv(BT,named['BC'])==1:
   assert 'arith.addf' not in ir
   assert not re.search(re.escape(dots[1][1])+r' = arith.constant',ir)
  else:
   assert 'arith.addf' in ir and len(re.findall(r'scf.for',ir))==2
 record={'case':label,'status':'compiled','T':T,'K':K,'V':V,'BT':BT,
   'want_seed':want_seed,'want_final':want_final,'wrapper_admitted':True,'grid':calls['grid'],
   'signature':signature,'constants':named,'scalar_values':expected,'options':options,
   'source':pin(SOURCE),'artifacts':artifacts,'metadata':compiled.metadata._asdict()}
 profiles.append(record);print(label,'compiled',flush=True)
report={'task':'chunk_delta_fwd_static_dot','status':'PASS','source':pin(SOURCE),
   'checker':pin(Path(__file__)),'triton_version':triton.__version__,'target':'SM80',
   'autotune_warps':configs,'compiled_num_warps':[1],'profiles':profiles,'GPU_execution':False,
   'harness_adaptations':'CPU-backed Tensor proxies adapt CUDA identity/context/allocation; actual wrapper guards and launch arguments feed the real JIT binder and packer.',
   'scope':'Seven actual candidate wrapper/JIT profiles; NC=1 live carry, NC>1 separate outer add, scalar FP32 FMA. No all-config or universal GPU resource guarantee.'}
path=OUT/'report.json';path.write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(pin(path)),flush=True)

"""Actual wrapper/JIT binding controls for static dot-tile admission; no GPU."""
from pathlib import Path
from types import SimpleNamespace
import argparse,contextlib,hashlib,importlib.util,inspect,json,os,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[4]
SOURCE=ROOT/'bench/repairs/tritonbench_g/chunk_delta_fwd/chunk_delta_fwd_repaired.py'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=ROOT/'.veritile-cache/kernel-evidence/chunk_delta_fwd/min-dot')
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
assert pin(SOURCE)['sha256']=='cb102b78b196326239b44aa330b78f76d46120997a2b152ca94c52d67e74e193'
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
for label,K,BT,T in [('key_width_one',1,16,16),('chunk_width_one',16,1,1),('legal_control',16,16,16)]:
 inputs=[Tensor(torch.ones((1,1,T,K),dtype=torch.float32)) for _ in range(2)]
 inputs.append(Tensor(torch.ones((1,1,T,1),dtype=torch.float32)))
 calls.clear();returned=m.chunk_fwd_h_fn_repaired(*inputs,BT)
 assert calls['grid']==(1,1,1)
 assert returned[0].shape==(1,1,triton.cdiv(T,BT)*K,1) and returned[1].shape==(1,1,T,1)
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
 record={'case':label,'K':K,'BT':BT,'T':T,'wrapper_admitted':True,'grid':calls['grid'],
         'signature':signature,'constants':named,'scalar_values':expected,'options':options,
         'correction_dot_shapes':[[named['BC'],named['BK']],[named['BK'],named['BV']]],
         'update_dot_shapes':[[named['BK'],named['BC']],[named['BC'],named['BV']]],'source':pin(SOURCE)}
 try:
  compiled=triton.compile(ASTSource(kernel,signature=signature,constexprs=constants,attrs=attrs),target=target,options=options)
 except Exception as error:
  record.update(status='rejected',diagnostic=str(error),exception=type(error).__name__)
  assert label!='legal_control' and 'Input shapes should have M >= 1, N >= 1 and K >= 16' in str(error),record
  site='b_v -= tl.dot' if label=='key_width_one' else 'b_h_cumsum += tl.dot'
  assert site in str(error),record
  record['rejected_dot']='correction' if label=='key_width_one' else 'update'
 else:
  assert label=='legal_control',record
  artifacts=[]
  for ext in ('ttir','ttgir','llir','ptx'):
   path=OUT/(label+'.'+ext);path.write_text(compiled.asm[ext]);artifacts.append(pin(path))
  assert 'fma.rn.f32' in compiled.asm['ptx'] and 'mma.sync' not in compiled.asm['ptx']
  record.update(status='compiled',artifacts=artifacts,metadata=compiled.metadata._asdict())
 profiles.append(record)
 print(label,record['status'],flush=True)
semantic=Path(inspect.getfile(triton.language.semantic.TritonSemantic.dot))
backend_source=Path(inspect.getfile(min_dot_size))
minimum=min_dot_size(target)(triton.language.block_type(triton.language.float32,[16,16]),triton.language.block_type(triton.language.float32,[16,16]))
assert minimum==(1,1,16)
minimum_source=inspect.getsource(min_dot_size)
assert 'num_warps' not in minimum_source
minimum_path=OUT/'nvidia-min-dot-size.py';minimum_path.write_text(minimum_source)
report={'task':'chunk_delta_fwd','status':'SOURCE_DEFECT_CONFIRMED','source':pin(SOURCE),'checker':pin(Path(__file__)),
        'triton_version':triton.__version__,'target':'SM80','autotune_warps':configs,'profiles':profiles,
        'semantic_source':pin(semantic),'backend_source':pin(backend_source),'minimum_shape':minimum,
        'minimum_rule_excerpt':pin(minimum_path),'compiled_num_warps':[configs[0]],
        'warp_independence':'The pinned NVIDIA minimum rule depends on operand bitwidth, not num_warps; FP32 reduction dimension must be at least 16.',
        'GPU_execution':False,
        'harness_adaptations':'CPU-backed Tensor proxies expose actual shape, strides, dtype, storage and pointer metadata while CUDA identity/context and fresh allocation are adapted. The canonical wrapper guards and launch expression execute unchanged; the real JIT binder receives the underlying CPU Tensor pointer arguments.',
        'scope':'Two admitted nonempty actual wrapper cases fail static dot shape admission before lowering; one same-config legal control compiles. No device resource or numerical experiment claim.'}
path=OUT/'report.json';path.write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(pin(path)),flush=True)

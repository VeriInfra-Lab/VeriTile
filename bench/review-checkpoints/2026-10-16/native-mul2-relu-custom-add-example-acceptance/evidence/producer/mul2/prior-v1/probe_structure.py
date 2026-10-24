"""Execute original wrapper AST with real CPU metadata and a launch stub."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,hashlib,json,random,types
import torch,triton
import triton.runtime.jit as jit
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
import importlib.util
W=Path(__file__).resolve().parent
SOURCE=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/triton_mul2/triton_mul2.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tree=ast.parse(SOURCE.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['triton_mul2','triton_mul2_inplace']]
events=[];allocation_fails=False
class Tensor:
 def __init__(self,value,label):self.value=value;self.label=label
 def numel(self):events.append(['numel',self.label,self.value.numel()]);return self.value.numel()
def zeros_like(x):
 events.append(['zeros_like',x.label])
 if allocation_fails:raise RuntimeError('injected allocation failure')
 return Tensor(torch.zeros_like(x.value),'output')
def cdiv(n,b):
 events.append(['cdiv',n,b]);return triton.cdiv(n,b)
class Kernel:
 def __init__(self,name):self.name=name
 def __getitem__(self,grid):
  events.append(['kernel_index',self.name,list(grid)])
  def call(*args):
   events.append(['launch_stub',self.name,[a.label if isinstance(a,Tensor) else a for a in args]])
  return call
env={'torch':types.SimpleNamespace(zeros_like=zeros_like),'triton':types.SimpleNamespace(cdiv=cdiv),'mul2_kernel':Kernel('mul2_kernel'),'mul2_inplace_kernel':Kernel('mul2_inplace_kernel')}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),env)
def metadata(t):
 return {'shape':list(t.shape),'strides':list(t.stride()),'numel':t.numel(),'dtype':str(t.dtype),'device':str(t.device),'storage_offset':t.storage_offset(),'storage_base':t.untyped_storage().data_ptr(),'storage_bytes':t.untyped_storage().nbytes(),'data_pointer':t.data_ptr(),'element_size':t.element_size(),'flat_prefix_available':t.storage_offset()*t.element_size()+t.numel()*t.element_size()<=t.untyped_storage().nbytes()}
base=torch.arange(120,dtype=torch.float32).reshape(10,12)
views={'scalar':torch.tensor(3.),'contiguous':base,'transpose':base.t(),'offset_dense':base[2:5],'sliced_holes':base[:,::2],'broadcast':base[:1,:1].expand(4,7),'empty':base[:0],'empty_transpose':base[:0].t(),'empty_strided_offset':base[7:7,::2]}
records=[]
for name,value in views.items():
 for wrapper in ['triton_mul2','triton_mul2_inplace']:
  for block in [0,-16,1,3,16,512,1024]:
   x=Tensor(value,'input');events=[]
   try:
    out=env[wrapper](x,block);status='stub_returned';returned=metadata(out.value)
    assert (out is x)==(wrapper=='triton_mul2_inplace')
    if wrapper=='triton_mul2':assert torch.count_nonzero(out.value).item()==0
   except Exception as e:status=type(e).__name__;returned=None
   assert events[0][0]==('zeros_like' if wrapper=='triton_mul2' else 'numel')
   assert (status=='ZeroDivisionError')==(block==0)
   records.append({'view':name,'wrapper':wrapper,'block':block,'status':status,'input':metadata(value),'returned':returned,'events':events[:]})
allocation_fails=True;events=[]
try:env['triton_mul2'](Tensor(base,'input'),0)
except RuntimeError:pass
else:raise AssertionError('allocation error not propagated')
assert events==[['zeros_like','input']]
allocation_order=events[:];allocation_fails=False
DTYPES=['bool','uint8','int8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
allocation=[]
for name in DTYPES:
 try:
  x=torch.empty((3,),dtype=getattr(torch,name));out=torch.zeros_like(x)
  allocation.append({'dtype':name,'status':'allocated','input':metadata(x),'output':metadata(out),'raw_output_bytes':list(out.untyped_storage())})
 except Exception as e:allocation.append({'dtype':name,'status':'error','exception_type':type(e).__name__,'message':str(e)})
# Actual binder for annotated n_elements and constexpr block arguments.
prefix=SOURCE.read_text().split('################################################################')[0];p=W/'evidence/source-prefix.py';p.write_text(prefix)
spec=importlib.util.spec_from_file_location('mul2_binder',p);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
backend=CUDABackend(GPUTarget('cuda',90,32));binder_records=[]
for name in ['mul2_kernel','mul2_inplace_kernel']:
 kernel=getattr(module,name);binder=jit.create_function_from_signature(kernel.signature,kernel.params,backend)
 for n in [0,1,2**31-1,2**31,2**32-1,2**32+1,2**63-1]:
  args=[jit.MockTensor(torch.float32)]*(2 if name=='mul2_kernel' else 1)+[n,16]
  params,specialization,kwargs=binder(*args);options,signature,constexprs,attrs=kernel._pack_args(backend,kwargs,params,specialization,kwargs)
  assert signature['n_elements']=='i32'
  binder_records.append({'kernel':name,'n':n,'signature':signature,'specialization':specialization,'constexprs':{str(k):v for k,v in constexprs.items()},'driver_int32_word':n%(2**32),'driver_int32_signed':(n+2**31)%(2**32)-2**31})
# Independent full tile load/store replay: in-place jobs must occur once.
rng=random.Random(2490);schedules=[]
for width in [1,2,4,8]:
 for n in [0,1,15,16,17,511,512,513,1024,1025]:
  for block in [1,16,512,1024]:
   for inplace in [False,True]:
    initial=bytearray(rng.randrange(256) for _ in range((2*n+64)*width));xstart=8*width;ystart=xstart if inplace else (n+24)*width
    read=lambda mem,p:int.from_bytes(mem[p:p+width],'little')
    multiply=lambda value:2*value%(1<<(8*width))
    reference=bytearray(initial)
    for i in range(n):reference[ystart+i*width:ystart+(i+1)*width]=multiply(read(initial,xstart+i*width)).to_bytes(width,'little')
    pids=list(range((n+block-1)//block));shuffled=pids[:];rng.shuffle(shuffled)
    for order in [pids,list(reversed(pids)),shuffled]:
     memory=bytearray(initial)
     for pid in order:
      tile=[(i,read(memory,xstart+i*width)) for lane in range(block) if (i:=pid*block+lane)<n]
      for i,value in tile:memory[ystart+i*width:ystart+(i+1)*width]=multiply(value).to_bytes(width,'little')
     assert memory==reference
     schedules.append({'width':width,'n':n,'block':block,'inplace':inplace,'program_count':len(order)})
assert (2*(2*3))%(2**32)!=2*3
report={'status':'passed','source_sha256':sha(SOURCE),'original_wrapper_ast_executed':True,'scope':'CPU metadata plus launch stubs and synthetic integer-byte schedules; no native numerical validation','gpu_executions':0,'wrapper_cases':records,'allocation_precedes_cdiv_error':allocation_order,'dtype_zeros_like':allocation,'annotated_n_binder':binder_records,'schedule_cases':schedules,'counts':{'wrapper_metadata_cases':len(records),'dtype_allocations':len(allocation),'binder_sizes':len(binder_records),'byte_schedule_cases':len(schedules)},'inplace_schedule_boundary':'Each active address is owned once per launch; repeated in-place programs can double again and are not accepted as equivalent schedules.'}
(W/'evidence/source-and-metadata-probe.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['counts']))

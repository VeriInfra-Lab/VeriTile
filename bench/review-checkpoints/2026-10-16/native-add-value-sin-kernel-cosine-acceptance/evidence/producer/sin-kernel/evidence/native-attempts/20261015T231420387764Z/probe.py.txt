"""Exact original source, all canonical dtype routes; offline CUDA compilation only."""
import ast,hashlib,importlib.util,inspect,json,os,sys,datetime
from pathlib import Path
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
O=W/'evidence/native-attempts'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes());os.environ['TRITON_CACHE_DIR']=str(O/'cache')
import torch,triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
spec=importlib.util.spec_from_file_location('sin_kernel_exact_'+stamp,W/'evidence/source-prefix.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);opts=backend.parse_options({});context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
records=[]
for name in names:
 row={'dtype':name,'output':name,'n_elements':'i32','BLOCK_SIZE':1024,'phase':'binder'}
 try:
  sig={'x_ptr':'*'+jit.type_canonicalisation_dict[name],'output_ptr':'*'+jit.type_canonicalisation_dict[name],'n_elements':'i32'};row['signature']=sig;row['phase']='frontend'
  src=ASTSource(mod.kernel_function,sig,constexprs={'BLOCK_SIZE':1024})
  module=src.make_ir(target,opts,backend.get_codegen_implementation(opts),backend.get_module_map(),context)
  p=O/(name+'.frontend.ttir');p.write_text(str(module));row['frontend']={'path':p.name,'sha256':sha(p)};row['phase']='backend'
  kernel=triton.compile(src,target=target);artifacts={}
  for ext,data in kernel.asm.items():
   p=O/(name+'.'+ext);p.write_bytes(data if isinstance(data,bytes) else str(data).encode());artifacts[p.name]=sha(p)
  row.update(status='compiled',artifacts=artifacts,metadata=kernel.metadata._asdict())
 except Exception as e:row.update(status='rejected',error_type=type(e).__name__,error=str(e))
 records.append(row)
binder=jit.create_function_from_signature(mod.kernel_function.signature,mod.kernel_function.params,backend);x=torch.empty(0,dtype=torch.float32);binding=[]
for n in [0,1,2,2**31-1,2**31,2**32-1,2**32,2**41-1024,2**41,2**63-1]:
 try:binding.append({'n':n,'binding_repr':repr(binder(x,x,n,1024))})
 except Exception as e:binding.append({'n':n,'error_type':type(e).__name__,'error':str(e)})
(O/'binder.json').write_text(json.dumps(binding,indent=2)+'\n');(O/'records.json').write_text(json.dumps(records,indent=2,default=str)+'\n')
(O/'jit-binder-source.txt').write_text(inspect.getsource(jit.create_function_from_signature))
for name in ['jit.py','driver.c']:
 p=Path('/usr/local/lib/python3.12/site-packages/triton')/('runtime/jit.py' if name=='jit.py' else 'backends/nvidia/driver.c');(O/name).write_bytes(p.read_bytes())
receipt={'source_prefix_sha256':sha(W/'evidence/source-prefix.py'),'torch':torch.__version__,'triton':triton.__version__,'target':vars(target),'options':vars(opts),'annotations':[(p.name,p.annotation,p.annotation_type,p.is_constexpr) for p in mod.kernel_function.params],'cases':len(records),'compiled':sum(r['status']=='compiled' for r in records),'gpu_executed':False,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');latest={'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')};(W/'evidence/native-latest.json').write_text(json.dumps(latest,indent=2)+'\n');print(json.dumps({'receipt':latest,'counts':{phase:sum(r['phase']==phase and r['status']=='rejected' for r in records) for phase in ['binder','frontend','backend']},'compiled':receipt['compiled']},indent=2))

"""Original binder/packer specializations at finite-index boundaries; no launch."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');O=W/'evidence/abi-profiles'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();os.environ['TRITON_CACHE_DIR']=str(O/'cache')
(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes())
import torch,triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
spec=importlib.util.spec_from_file_location('sin_abi_'+stamp,W/'evidence/source-prefix.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);binder=jit.create_function_from_signature(mod.kernel_function.signature,mod.kernel_function.params,backend)
extents=[0,1,2,1023,1024,1025,2**31-1,2**31,2**31+1,2**32-1,2**32,2**32+1,(2**31-1)*1024,(2**31-1)*1024+1,2**63-1];rows=[]
for name in ['float32','float64']:
 for n in extents:
  params,specialization,options=binder(jit.MockTensor(getattr(torch,name)),jit.MockTensor(getattr(torch,name)),n,BLOCK_SIZE=1024)
  options,signature,constexprs,attrs=mod.kernel_function._pack_args(backend,options,params,specialization,options)
  assert signature['n_elements']=='i32' and constexprs=={(3,):1024}
  src=ASTSource(mod.kernel_function,signature,constexprs,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
  raw=str(src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context));tag=name+'_'+str(n);p=O/(tag+'.raw.ttir');p.write_text(raw)
  kernel=triton.compile(src,target=target,options=vars(options));artifacts={}
  for ext,data in kernel.asm.items():
   p=O/(tag+'.'+ext);p.write_bytes(data if isinstance(data,bytes) else str(data).encode());artifacts[p.name]=sha(p)
  rows.append({'dtype':name,'n':n,'signature':signature,'specialization':specialization,'constexprs':{str(k):v for k,v in constexprs.items()},'attrs':{str(k):v for k,v in attrs.items()},'raw_ttir_sha256':sha(O/(tag+'.raw.ttir')),'artifacts':artifacts})
(O/'records.json').write_text(json.dumps(rows,indent=2)+'\n');receipt={'status':'passed','source_prefix_sha256':sha(W/'evidence/source-prefix.py'),'profiles':len(rows),'extents':extents,'target':vars(target),'triton':triton.__version__,'gpu_executed':False,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(O/'receipt.json'),'sha256':sha(O/'receipt.json'),'profiles':len(rows)}))

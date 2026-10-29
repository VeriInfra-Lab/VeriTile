"""Unchanged source AST and offline CUDA frontend; no GPU launch."""
import ast,hashlib,importlib.util,inspect,json,os,sys,datetime
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT=ROOT/'evidence/native-attempts'/STAMP;OUT.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(OUT/'probe.py.txt').write_bytes(Path(__file__).read_bytes())
os.environ['TRITON_CACHE_DIR']=str(OUT/'cache')
import torch,triton,triton.language as tl
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
PREFIX=ROOT/'evidence/source-prefix.py'
spec=importlib.util.spec_from_file_location('add_example_exact_source_'+STAMP,PREFIX)
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);opts=backend.parse_options({})
context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
records=[]
for xn in names:
 for yn in names:
  record={'x':xn,'y':yn,'output':xn,'n_elements':'i32','BLOCK_SIZE':4}
  try:
   sig={'in_ptr0':'*'+jit.type_canonicalisation_dict[xn],'in_ptr1':'*'+jit.type_canonicalisation_dict[yn],'out_ptr':'*'+jit.type_canonicalisation_dict[xn],'n_elements':'i32'}
   record['signature']=sig
   src=ASTSource(mod.add_kernel,sig,constexprs={'BLOCK_SIZE':4})
   module=src.make_ir(target,opts,backend.get_codegen_implementation(opts),backend.get_module_map(),context)
   path=OUT/(xn+'__'+yn+'.ttir');path.write_text(str(module))
   record.update(status='frontend_generated',ttir=path.name,ttir_sha256=sha(path))
  except Exception as e:record.update(status='rejected',error_type=type(e).__name__,error=str(e))
  records.append(record)
# Exact binder behavior for forced i32 annotated n. CPU tensors only.
binder=jit.create_function_from_signature(mod.add_kernel.signature,mod.add_kernel.params,backend)
x=torch.empty(0,dtype=torch.float32)
binding=[]
for n in [0,1,2,2**31-1,2**31,2**32-1,2**32,2**63-1]:
 try:
  value=binder(x,x,x,n,4)
  binding.append({'n':n,'binding_repr':repr(value)})
 except Exception as e:binding.append({'n':n,'error_type':type(e).__name__,'error':str(e)})
(OUT/'binder.json').write_text(json.dumps(binding,indent=2)+'\n')
(OUT/'records.json').write_text(json.dumps(records,indent=2)+'\n')
(OUT/'jit-binder-source.txt').write_text(inspect.getsource(jit.create_function_from_signature))
summary={'source_prefix_sha256':sha(PREFIX),'probe_sha256':sha(Path(__file__)),'torch':torch.__version__,'triton':triton.__version__,'target':vars(target),'options':vars(opts),'annotations':[(p.name,p.annotation,p.annotation_type,p.is_constexpr) for p in mod.add_kernel.params],'canonical_names':names,'cases':len(records),'frontend_generated':sum(r['status']=='frontend_generated' for r in records),'rejected':sum(r['status']=='rejected' for r in records),'gpu_executed':False,'native_backend_compiled':False,'artifacts':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}}
(OUT/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
(ROOT/'evidence/native-latest.json').write_text(json.dumps({'receipt':str((OUT/'receipt.json').relative_to(ROOT)),'sha256':sha(OUT/'receipt.json')},indent=2)+'\n')
print(json.dumps({'receipt':str(OUT/'receipt.json'),'sha256':sha(OUT/'receipt.json'),'cases':summary['cases'],'frontend_generated':summary['frontend_generated'],'rejected':summary['rejected'],'binding':binding},indent=2),flush=True)

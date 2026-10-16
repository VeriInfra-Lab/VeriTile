"""Original vector kernel: all 22 dense Torch signatures, forced-i32 ABI, sm90.
No device launch. Each attempt preserves stdout/stderr and every IR stage.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,os,traceback
W=Path(__file__).resolve().parent
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
O=W/'evidence/native-attempts'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes())
os.environ['TRITON_CACHE_DIR']=str(O/'cache')
os.environ['TMPDIR']=str(O/'tmp');(O/'tmp').mkdir()
source=R/'bench/tritonbench_g/vector_addition_custom/vector_addition_custom.py'
original=source.read_text();prefix=original.split('################################################################')[0]
asts=lambda s:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name in ['_add_kernel','custom_add']}
assert asts(original)==asts(prefix)
(O/'source-prefix.py').write_text(prefix)
import torch,triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
spec=importlib.util.spec_from_file_location('vector_revision1_'+stamp,O/'source-prefix.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
binder=jit.create_function_from_signature(mod._add_kernel.signature,mod._add_kernel.params,backend)
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
records=[]
for xn in names:
 for yn in names:
  row={'x':xn,'y':yn,'output':xn,'size':2**31,'BLOCK':16};records.append(row)
  try:
   args=[jit.MockTensor(getattr(torch,n)) for n in [xn,yn,xn]]+[2**31]
   params,specialization,options=binder(*args,BLOCK=16)
   options,signature,constexprs,attrs=mod._add_kernel._pack_args(backend,options,params,specialization,options)
   row.update(signature=signature,specialization=specialization,constexprs={str(k):v for k,v in constexprs.items()})
   assert signature['size']=='i32'
   src=ASTSource(mod._add_kernel,signature,constexprs,attrs)
   context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
   raw=src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
   p=O/(xn+'__'+yn+'.raw.ttir');p.write_text(str(raw));row['raw_sha256']=sha(p);row['frontend']='generated'
  except Exception as e:
   row.update(frontend='rejected',error_type=type(e).__name__,error=str(e));continue
  try:
   compiled=triton.compile(src,target=target,options=vars(options));artifacts={}
   for ext,data in compiled.asm.items():
    p=O/(xn+'__'+yn+'.'+ext);p.write_bytes(data if isinstance(data,bytes) else str(data).encode());artifacts[p.name]=sha(p)
   row.update(backend='compiled',artifacts=artifacts)
  except Exception as e:row.update(backend='rejected',error_type=type(e).__name__,error=str(e))
  print(xn,yn,row.get('frontend'),row.get('backend'),flush=True)
(O/'records.json').write_text(json.dumps(records,indent=2)+'\n')
summary={'original_source':str(source),'original_source_sha256':sha(source),'production_AST_unchanged':True,'torch':torch.__version__,'triton':triton.__version__,'target':vars(target),'canonical_names':names,'cases':len(records),'frontend_generated':sum(r['frontend']=='generated' for r in records),'frontend_rejected':sum(r['frontend']=='rejected' for r in records),'backend_compiled':sum(r.get('backend')=='compiled' for r in records),'backend_rejected':sum(r.get('backend')=='rejected' for r in records),'gpu_launches':0,'large_allocations':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
(W/'evidence/native-latest.json').write_text(json.dumps({'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')},indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='artifacts'},indent=2))

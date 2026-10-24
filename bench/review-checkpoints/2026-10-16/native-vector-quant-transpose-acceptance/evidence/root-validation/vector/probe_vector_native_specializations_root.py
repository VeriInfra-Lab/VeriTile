from pathlib import Path
import ast, hashlib, importlib.util, json, os, sys, traceback
sys.dont_write_bytecode=True
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
P=Path('/tmp/veritile-standardization-20261015/vector-finite-acceptance-revision2-wip')
V=Path('/tmp/veritile-standardization-20261015/vector-native-revision2-root-validation/specialization-probe')
assert not V.exists();V.mkdir(parents=True)
os.environ['TRITON_CACHE_DIR']=str(V/'cache')
os.environ['TMPDIR']=str(V/'tmp');(V/'tmp').mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
source=R/'bench/tritonbench_g/vector_addition/vector_addition.py'
original=source.read_text();prefix=original.split('################################################################')[0]
functions=lambda text:[ast.dump(n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name in {'add_kernel','add'}]
assert functions(original)==functions(prefix)
(V/'source.py').write_text(prefix)
import torch,triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
spec=importlib.util.spec_from_file_location('vector_root_specialization',V/'source.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target)
kernel=module.add_kernel;binder=jit.create_function_from_signature(kernel.signature,kernel.params,backend)
pairs=[('float8_e4m3fn','float8_e4m3fn'),('float8_e5m2','float8_e5m2'),
 ('float64','float8_e4m3fn'),('float64','float8_e5m2'),
 ('float8_e4m3fn','float64'),('float8_e5m2','float64')]
rows=[]
for x,y in pairs:
 for n in [0,1,2049]:
  name=f'{x}__{y}__n{n}';row={'x':x,'y':y,'n':n,'BLOCK_SIZE':1024};stage='binder'
  try:
   args=[jit.MockTensor(getattr(torch,d)) for d in [x,y,x]]+[n]
   params,specialization,kwargs=binder(*args,BLOCK_SIZE=1024)
   options,signature,constexprs,attrs=kernel._pack_args(backend,kwargs,params,specialization,kwargs)
   row.update(signature=signature,specialization=specialization,constexprs={str(k):v for k,v in constexprs.items()},attrs={str(k):v for k,v in attrs.items()},options=vars(options))
   stage='frontend';src=ASTSource(kernel,signature,constexprs,attrs)
   context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
   raw=src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context)
   (V/(name+'.raw.ttir')).write_text(str(raw));stage='backend'
   compiled=triton.compile(src,target=target,options=vars(options))
   for ext in ['ttir','ttgir','llir','ptx']:(V/(name+'.'+ext)).write_text(compiled.asm[ext])
   row['status']='compiled'
  except Exception:
   row.update(status='rejected',failure_stage=stage,exception=traceback.format_exc())
  (V/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
  print(name,row['status'],row.get('failure_stage'),flush=True)
receipt={'kind':'root_offline_backend_specialization_review','source_sha256':sha(source),
 'production_AST_unchanged':True,'producer_receipt_sha256':sha(P/'evidence/receipt.json'),
 'target':vars(target),'torch_version':torch.__version__,'triton_version':triton.__version__,
 'cases':len(rows),'all_expected_backend_rejections':all(r.get('failure_stage')=='backend' for r in rows),
 'records':rows,'gpu_launches':0,'large_allocations':0,
 'artifacts':{p.name:sha(p) for p in sorted(V.iterdir()) if p.is_file()}}
(V/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in {'records','artifacts'}},indent=2))
assert receipt['all_expected_backend_rejections']

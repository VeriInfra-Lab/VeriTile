"""Independent FP32/BF16-only compiler and host boundary replay; no GPU launch."""
from pathlib import Path
import ast,ctypes,hashlib,importlib.util,json,math,os,re,struct,sys,traceback
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;P=W.parent/'cosine-native-finite-acceptance-revision1-wip';R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
assert sha(P/'evidence/receipt.json')==(W/'producer-pin.txt').read_text().strip()
O=W/'focused-native-001';O.mkdir();os.environ['TRITON_CACHE_DIR']=str(O/'cache')
source=R/'bench/tritonbench_g/cosine_compute/cosine_compute.py';text=source.read_text();prefix=text.split('################################################################')[0]
nodes=lambda x:[ast.dump(n) for n in ast.parse(x).body if isinstance(n,ast.FunctionDef) and n.name in ['cos_func','cos']]
assert nodes(text)==nodes(prefix);(O/'source.py').write_text(prefix)
import torch,triton
import triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
spec=importlib.util.spec_from_file_location('independent_cosine_source',O/'source.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);kernel=m.cos_func;binder=jit.create_function_from_signature(kernel.signature,kernel.params,backend)
rows=[];extents=[1,4096,2**31,2**40,2**40+1,2**52+1,2**52+2,2**63-1]
for name in ['float32','bfloat16']:
 for n in extents:
  block=triton.next_power_of_2(math.ceil(math.sqrt(n)));tag=name+'-'+str(n);row={'dtype':name,'n':n,'block':block,'grid':triton.cdiv(n,block)}
  try:
   params,specialization,options=binder(jit.MockTensor(getattr(torch,name)),jit.MockTensor(getattr(torch,name)),n,block)
   options,signature,constexprs,attrs=kernel._pack_args(backend,options,params,specialization,options)
   assert signature['n_elements']=='i32' and constexprs=={(3,):block};row.update(signature=signature,constexprs={str(k):v for k,v in constexprs.items()},specialization=specialization)
   src=ASTSource(kernel,signature,constexprs,attrs);context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
   raw=str(src.make_ir(target,options,backend.get_codegen_implementation(options),backend.get_module_map(),context));(O/(tag+'.raw.ttir')).write_text(raw)
   assert block<=2**20;row.update(status='frontend_passed')
   if n==4096:
    compiled=triton.compile(src,target=target,options=vars(options));row['backend']='compiled'
    for ext,data in compiled.asm.items():(O/(tag+'.'+ext)).write_bytes(data if isinstance(data,bytes) else str(data).encode())
  except Exception as error:
   row.update(status='rejected',error_type=type(error).__name__,exception=traceback.format_exc())
   assert block>2**20,(name,n,row['exception'])
   assert ('cannot be represented' in str(error) or 'out of range' in str(error) or 'int32' in str(error)) if block>=2**31 else ('tensor' in str(error).lower() or 'numel' in str(error).lower())
  rows.append(row)
# Compare only the two independently rebuilt default profiles. Path/debug metadata
# are normalized explicitly; emitted artifacts remain available unchanged.
ref=load(P/'evidence/native-latest.json');old=(P/ref['receipt']).parent
same=[];debug=[]
for name in ['float32','bfloat16']:
 for ext in ['raw.ttir','ttir','ttgir','llir','ptx']:
  a=(old/(name+'.'+ext)).read_text();b=(O/(name+'-4096.'+ext)).read_text()
  a=a.replace(str(old),'<RUN>');b=b.replace(str(O),'<RUN>')
  if ext=='ptx':
   def executable(x):
    match=re.search(r'^\s*\.section\s+\.debug',x,re.M)
    if match:
     assert not re.search(r'^\s*(?:\.visible\s+)?(?:\.entry|\.func)',x[match.start():],re.M);return x[:match.start()]
    return x
   if a!=b:debug.append(name)
   a,b=executable(a),executable(b)
  assert a==b,(name,ext);same.append(name+'.'+ext)
# Independent small host boundary sample, exactly stated, not a global conformance proof.
inputs=[0,1,2,3,4,5,4095,4096,4097,2**31-1,2**31,2**31+1,2**40,2**40+1,2**52,2**52+1,2**52+2,2**53-1,2**53,2**53+1,2**53+2,2**53+3,2**61,2**63-1]
mathrows=[]
for n in inputs:
 value=math.sqrt(n);integer=int(float(n));ceil=math.ceil(value);block=triton.next_power_of_2(ceil)
 mathrows.append({'n':n,'integer64':integer,'sqrt_bits':int.from_bytes(struct.pack('>d',value),'big'),'sqrt_hex':value.hex(),'ceil':ceil,'block':block,'grid':triton.cdiv(n,block) if block else None})
assert mathrows[15]['n']==2**52+1 and mathrows[15]['block']==2**26
assert mathrows[16]['block']==2**27
(O/'frontend-records.json').write_text(json.dumps(rows,indent=2)+'\n');(O/'host-boundaries.json').write_text(json.dumps(mathrows,indent=2)+'\n')
report={'status':'passed','source_sha256':sha(source),'producer_receipt_sha256':sha(P/'evidence/receipt.json'),'dtype_subset':['float32','bfloat16'],'frontend_profiles':len(rows),'backend_compiles':2,'equal_IR_artifacts':same,'PTX_debug_normalization':debug,'host_boundary_samples':len(inputs),'native_host_conformance':'not_proved; sampled current environment only','fegetround':ctypes.CDLL(None).fegetround(),'torch':torch.__version__,'triton':triton.__version__,'target':vars(target),'GPU_launches':0,'large_allocations':0,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='artifacts'}))

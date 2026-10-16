from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');O=W/'evidence/native-attempts'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes());os.environ['TRITON_CACHE_DIR']=str(O/'cache')
import torch,triton,triton.runtime.jit as jit
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia.compiler import CUDABackend
from triton._C.libtriton import ir
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py');tree=ast.parse(R.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))][:3]+[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['relu_kernel','relu']];p=O/'exact_prefix.py';p.write_text(ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n')
for n in nodes:
 if isinstance(n,ast.FunctionDef):assert ast.dump(n,include_attributes=False)==ast.dump(next(t for t in ast.parse(p.read_text()).body if isinstance(t,ast.FunctionDef) and t.name==n.name),include_attributes=False)
spec=importlib.util.spec_from_file_location('relu_exact_'+stamp,p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
target=GPUTarget('cuda',90,32);backend=CUDABackend(target);opts=backend.parse_options({});context=ir.context();ir.load_dialects(context);backend.load_dialects(context)
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
cases=[('dtype-'+name,name,2**31+1) for name in names]+[('extent-'+str(n),'float32',n) for n in [0,1,2,2**31-1,2**31,2**32-1,2**32,2**63-1,2**63,2**64-1,2**64]]
records=[]
for name,dtype,n in cases:
 row={'case':name,'dtype':dtype,'N':n,'block_size':1024,'grid_x':(n+1023)//1024}
 try:
  sig={'x_ptr':'*'+jit.type_canonicalisation_dict[dtype],'out_ptr':'*fp32'};row['signature']=sig;src=ASTSource(mod.relu_kernel,sig,constexprs={'N':n,'block_size':1024});m=src.make_ir(target,opts,backend.get_codegen_implementation(opts),backend.get_module_map(),context);q=O/(name+'.raw.ttir');q.write_text(str(m));row.update(frontend='generated',raw_ttir=q.name,raw_ttir_sha256=sha(q))
  try:
   k=triton.compile(src,target=target);row['native']='compiled';row['artifacts']={}
   for ext,data in k.asm.items():
    q=O/(name+'.'+ext);q.write_bytes(data if isinstance(data,bytes) else str(data).encode());row['artifacts'][q.name]=sha(q)
  except Exception as e:row.update(native='rejected',native_error_type=type(e).__name__,native_error=str(e))
 except Exception as e:row.update(frontend='rejected',error_type=type(e).__name__,error=str(e))
 records.append(row)
binder=jit.create_function_from_signature(mod.relu_kernel.signature,mod.relu_kernel.params,backend);x=torch.empty(0);binding=[]
for n in [0,1,2**31+1,2**32,2**63-1,2**63,2**64]:
 try:binding.append({'N':n,'binding':repr(binder(x,x,n,1024))})
 except Exception as e:binding.append({'N':n,'error':str(e),'error_type':type(e).__name__})
(O/'binder.json').write_text(json.dumps(binding,indent=2)+'\n');(O/'records.json').write_text(json.dumps(records,indent=2)+'\n');receipt={'source':str(R),'source_sha256':sha(R),'source_ast_identity':True,'torch':torch.__version__,'triton':triton.__version__,'target':vars(target),'names':names,'gpu_executed':False,'native_backend_attempted':True,'cases':len(records),'frontend_generated':sum(r['frontend']=='generated' for r in records),'native_compiled':sum(r.get('native')=='compiled' for r in records),'artifacts':{q.name:sha(q) for q in O.iterdir() if q.is_file()}};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(W/'evidence/native-latest.json').write_text(json.dumps({'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')},indent=2)+'\n');print(json.dumps({'receipt':str(O/'receipt.json'),'sha256':sha(O/'receipt.json'),'cases':len(records),'frontend_generated':receipt['frontend_generated'],'native_compiled':receipt['native_compiled']},indent=2))

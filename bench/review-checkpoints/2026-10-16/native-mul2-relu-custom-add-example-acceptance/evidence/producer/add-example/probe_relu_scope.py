from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');O=W/'evidence/relu-scope-review'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes());os.environ['TRITON_CACHE_DIR']=str(O/'cache')
import torch,triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
R=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/relu_triton_kernel/relu_triton_kernel.py');tree=ast.parse(R.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))][:3]+[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['relu_kernel','relu']];prefix=ast.Module(body=nodes,type_ignores=[]);p=O/'exact_prefix.py';p.write_text(ast.unparse(prefix)+'\n')
for n in nodes:
 if isinstance(n,ast.FunctionDef):assert ast.dump(n,include_attributes=False)==ast.dump(next(t for t in ast.parse(p.read_text()).body if isinstance(t,ast.FunctionDef) and t.name==n.name),include_attributes=False)
spec=importlib.util.spec_from_file_location('relu_review_'+stamp,p);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
records=[]
for n in [2**31,2**31+1,2**32-1,2**32,2**33]:
 row={'N':n,'block_size':1024,'grid_x':(n+1023)//1024,'grid_x_less_than_2p31':(n+1023)//1024<2**31}
 try:
  src=ASTSource(mod.relu_kernel,{'x_ptr':'*fp32','out_ptr':'*fp32'},constexprs={'N':n,'block_size':1024});k=triton.compile(src,target=GPUTarget('cuda',90,32));row['status']='compiled';row['artifacts']={}
  for ext,data in k.asm.items():
   q=O/(str(n)+'.'+ext);q.write_bytes(data if isinstance(data,bytes) else str(data).encode());row['artifacts'][q.name]=sha(q)
 except Exception as e:row.update(status='rejected',error_type=type(e).__name__,error=str(e))
 records.append(row)
(O/'records.json').write_text(json.dumps(records,indent=2)+'\n');receipt={'source':str(R),'source_sha256':sha(R),'source_ast_identity':True,'torch':torch.__version__,'triton':triton.__version__,'gpu_executed':False,'native_backend_compiled':True,'records':records,'artifacts':{q.name:sha(q) for q in O.iterdir() if q.is_file()}};(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(O/'receipt.json'),'sha256':sha(O/'receipt.json'),'records':[{k:v for k,v in r.items() if k!='artifacts'} for r in records]},indent=2))

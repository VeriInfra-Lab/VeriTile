from pathlib import Path
import datetime,hashlib,importlib.util,json,os,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');O=W/'evidence/backend-attempts'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes());os.environ['TRITON_CACHE_DIR']=str(O/'cache')
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
spec=importlib.util.spec_from_file_location('backend_exact_'+stamp,W/'evidence/source-prefix.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
latest=json.loads((W/'evidence/native-latest.json').read_text());front=W/latest['receipt'];records=json.loads((front.parent/'records.json').read_text());target=GPUTarget('cuda',90,32);out=[]
failed={(r['x'],r['y']) for r in json.loads((W/'evidence/backend-attempts/20261015T215542684768Z/records.json').read_text()) if r['status']=='rejected'} if os.environ.get('ADDEXAMPLE_RECHECK_FAILED_ONLY') else None
for row in records:
 if row['status']!='frontend_generated':continue
 if failed is not None and (row['x'],row['y']) not in failed:continue
 item={'x':row['x'],'y':row['y'],'signature':row['signature']}
 try:
  kernel=triton.compile(ASTSource(mod.add_kernel,row['signature'],constexprs={'BLOCK_SIZE':4}),target=target)
  paths={}
  for ext,data in kernel.asm.items():
   p=O/(row['x']+'__'+row['y']+'.'+ext);p.write_bytes(data if isinstance(data,bytes) else str(data).encode());paths[p.name]=sha(p)
  item.update(status='compiled',artifacts=paths,metadata=kernel.metadata._asdict())
 except Exception as e:item.update(status='rejected',error_type=type(e).__name__,error=str(e))
 out.append(item)
(O/'records.json').write_text(json.dumps(out,indent=2,default=str)+'\n');receipt={'original_source_prefix_sha256':sha(W/'evidence/source-prefix.py'),'frontend_receipt':latest,'target':vars(target),'triton':triton.__version__,'native_backend_compiled':True,'gpu_executed':False,'cases':len(out),'compiled':sum(r['status']=='compiled' for r in out),'rejected':sum(r['status']=='rejected' for r in out),'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(O/'receipt.json'),'sha256':sha(O/'receipt.json'),'cases':len(out),'compiled':receipt['compiled'],'rejected':receipt['rejected']},indent=2),flush=True)

"""Execute the original host prefix with recorded kernel calls; CPU tensors only."""
from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
W=Path(__file__).resolve().parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');O=W/'evidence/host-attempts'/stamp;O.mkdir(parents=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();(O/'probe.py.txt').write_bytes(Path(__file__).read_bytes())
import torch
spec=importlib.util.spec_from_file_location('sin_host_'+stamp,W/'evidence/source-prefix.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
def meta(x):return {'shape':list(x.shape),'strides':list(x.stride()),'numel':x.numel(),'dtype':str(x.dtype),'storage_offset':x.storage_offset(),'data_ptr':x.data_ptr(),'storage_base':x.untyped_storage().data_ptr(),'storage_bytes':x.untyped_storage().nbytes(),'device':str(x.device)}
calls=[]
class Recorder:
 def __getitem__(self,grid):
  def launch(x,out,n,**kw):calls.append({'grid':grid(kw),'n':n,'block':kw['BLOCK_SIZE'],'x':meta(x),'output':meta(out),'output_id':id(out)})
  return launch
mod.kernel_function=Recorder()
names=['bool','int8','uint8','int16','uint16','int32','uint32','int64','uint64','float16','bfloat16','float32','float64','float8_e4m3fn','float8_e5m2','float8_e4m3fnuz','float8_e5m2fnuz','complex32','complex64','complex128','float8_e8m0fnu','float4_e2m1fn_x2']
records=[]
for name in names:
 for n in [0,1,5]:
  row={'dtype':name,'size':n}
  try:
   x=torch.empty(n,dtype=getattr(torch,name));out=mod.call_kernel(x);assert id(out)==calls[-1]['output_id'];row.update(status='returned',x=meta(x),output=meta(out),launch=calls[-1])
  except Exception as e:row.update(status='error',error_type=type(e).__name__,error=str(e))
  records.append(row)
views={'scalar':torch.empty(()),'transpose':torch.empty(2,3).T,'offset':torch.empty(16)[3:7],'sliced':torch.empty(30)[3:23:2],'broadcast':torch.empty(1).expand(5),'empty_stride':torch.empty_strided((0,3),(7,2)),'empty_transpose':torch.empty(3,0).T,'empty_huge_shape':torch.empty(2**35,0),'empty_huge_offset':torch.empty(4).as_strided((0,),(1,),storage_offset=1000000)}
for name,x in views.items():
 out=mod.call_kernel(x);assert id(out)==calls[-1]['output_id'];records.append({'case':name,'status':'returned','x':meta(x),'output':meta(out),'launch':calls[-1]})
# Original AST wrapper with no tensor allocations: observable query/allocation/launch order.
tree=ast.parse((W/'evidence/source-prefix.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='call_kernel');code=compile(ast.Module(body=[fn],type_ignores=[]),'original-call_kernel','exec');traces=[]
for n in [0,1,1023,1024,1025,2**31,2**32+1,2**41,2**63-1]:
 for fail in [False,True]:
  trace=[]
  class X:
   def numel(self):trace.append(['numel',n]);return n
  x=X();out=object()
  class Torch:
   @staticmethod
   def empty_like(t):
    assert t is x;trace.append(['empty_like'])
    if fail:raise RuntimeError('instrumented allocation failure')
    return out
  class Kernel:
   def __getitem__(self,grid):
    trace.append(['grid_getitem'])
    def launch(t,o,extent,**meta):assert t is x and o is out and extent==n;trace.append(['launch',grid(meta),extent,meta])
    return launch
  class Triton:
   @staticmethod
   def cdiv(a,b):return (a+b-1)//b
  env={'torch':Torch,'triton':Triton,'kernel_function':Kernel()};exec(code,env)
  try:ret=env['call_kernel'](x);assert ret is out;trace.append(['return_same_output'])
  except RuntimeError as e:assert fail;trace.append(['allocation_error',str(e)])
  assert trace[0]==['numel',n] and trace[1]==['empty_like'];traces.append({'n':n,'allocation_failure':fail,'trace':trace})
(O/'records.json').write_text(json.dumps(records,indent=2)+'\n');(O/'ast-traces.json').write_text(json.dumps(traces,indent=2)+'\n')
receipt={'source_prefix_sha256':sha(W/'evidence/source-prefix.py'),'torch':torch.__version__,'cases':len(records),'ast_cases':len(traces),'cpu_metadata_only':True,'fake_launch_records_only':True,'gpu_executed':False,'artifacts':{p.name:sha(p) for p in O.iterdir() if p.is_file()}}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');latest={'receipt':str((O/'receipt.json').relative_to(W)),'sha256':sha(O/'receipt.json')};(W/'evidence/host-latest.json').write_text(json.dumps(latest,indent=2)+'\n');print(json.dumps({'receipt':latest,'cases':len(records),'ast_cases':len(traces),'errors':[r for r in records if r['status']=='error']},indent=2))

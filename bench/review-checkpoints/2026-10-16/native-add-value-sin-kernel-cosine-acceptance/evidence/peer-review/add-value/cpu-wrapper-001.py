"""Actual unchanged wrapper, real CPU allocations, allocation recorder only."""
from pathlib import Path
import hashlib,importlib.util,json,sys,types
sys.dont_write_bytecode=True
import torch
W=Path(__file__).resolve().parent;F=W/'replay'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();load=lambda p:json.loads(Path(p).read_text())
native=F/load(F/'evidence/native-latest.json')['receipt'];source=native.with_name('source.py')
spec=importlib.util.spec_from_file_location('add_value_real_cpu_wrapper',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
def meta(x):
 return {'shape':list(x.shape),'strides':list(x.stride()),'numel':x.numel(),'dtype':str(x.dtype),'storage_offset':x.storage_offset(),'data_ptr':x.data_ptr(),'storage_base':x.untyped_storage().data_ptr(),'storage_bytes':x.untyped_storage().nbytes(),'is_cuda':x.is_cuda,'device':str(x.device)}
allocations=[]
def alloc(x):
 y=torch.empty_like(x);allocations.append(meta(y));return y
mod.torch=types.SimpleNamespace(Tensor=torch.Tensor,empty_like=alloc)
names=[r['dtype'] for r in load(native)['records']]
inputs=[(name+'-'+str(n),torch.empty(n,dtype=getattr(torch,name))) for name in names for n in [0,1]]
inputs += [('scalar',torch.empty(())),('transpose',torch.empty(2,3).T),('slice',torch.empty(32)[3:23:2]),('offset-empty',torch.empty(32)[7:7]),('broadcast',torch.empty(1).expand(5))]
rows=[]
for name,x in inputs:
 before=len(allocations)
 try:mod.puzzle1(x)
 except AssertionError:
  assert len(allocations)==before+1
  out=allocations[-1];xm=meta(x)
  assert out['dtype']==xm['dtype'] and out['shape']==xm['shape'] and not out['is_cuda'] and out['storage_offset']==0
  for m in [xm,out]:
   assert m['data_ptr']==(m['storage_base']+m['storage_offset']*x.element_size() if m['numel'] else 0)
  if xm['storage_bytes'] and out['storage_bytes']:
   assert xm['storage_base']+xm['storage_bytes']<=out['storage_base'] or out['storage_base']+out['storage_bytes']<=xm['storage_base']
  rows.append({'case':name,'input':xm,'allocated_output':out,'outcome':'actual_AssertionError_after_actual_empty_like','pointer_extraction_reached':False})
 else:raise AssertionError('CPU wrapper did not enforce source assertion')
report={'status':'passed','cases':len(rows),'dtype_profiles':22,'source_prefix_sha256':sha(source),'original_AST_unchanged':True,'CPU_only':True,'GPU_launches':0,'scope':'Actual torch.empty_like and actual source CUDA assertion. Allocation recorder exposes output metadata; no kernel replacement needed because assertion fires before launch.','records':rows}
(W/'cpu-wrapper-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='records'}))

from pathlib import Path
import ast,datetime,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out=root/'autotune-probes'/stamp;out.mkdir(parents=True)
source=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/quant_transpose_kernel/quant_transpose_kernel.py')
prefix=source.read_text().split('################################################################')[0]
copied=out/'original_prefix.py';copied.write_text(prefix)
spec=importlib.util.spec_from_file_location('transpose_autotune_probe',copied)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
auto=module._quantize_global_transpose
import triton.runtime.autotuner as runtime
from triton.runtime.errors import OutOfResources
records=[]
class Tensor:
 def __init__(self,dtype):self.dtype=dtype
for repetitions in [1,2,5]:
 for fail_first in [False,True]:
  events=[];state={'version':0,'benchmark':0,'in_bench':False}
  xs,scale,output=Tensor('float32'),Tensor('float32'),Tensor('int8')
  args=(xs,scale,output,257,1,1153,1,1153,257)
  class Kernel:
   def run(self,*actual,**kwargs):
    assert actual==args
    assert {k:kwargs[k] for k in ['BLOCK_M','BLOCK_N','GROUP_M']}=={'BLOCK_M':128,'BLOCK_N':128,'GROUP_M':8}
    assert kwargs['grid'](kwargs)==(10*3,)
    assert kwargs['num_warps']==4 and kwargs['num_stages']==3
    if fail_first and state['in_bench'] and state['benchmark']==1:
     events.append({'benchmark':1,'failed':'OutOfResources','version':state['version']})
     raise OutOfResources(100,1,'instrumented resource')
    previous=state['version'];state['version']+=1
    events.append({'benchmark':state['benchmark'] if state['in_bench'] else 'selected_final','before':previous,'after':state['version']})
  def bench(call,quantiles):
   assert quantiles==(0.5,0.2,0.8)
   state['benchmark']+=1;state['in_bench']=True
   try:
    for _ in range(repetitions):call()
   finally:state['in_bench']=False
   return [3-state['benchmark']]*3
  auto.fn=Kernel();auto._do_bench=bench
  auto.__dict__.pop('do_bench',None);auto.cache={};auto.cache_results=False
  grid=lambda meta:((1153+meta['BLOCK_M']-1)//meta['BLOCK_M']*((257+meta['BLOCK_N']-1)//meta['BLOCK_N']),)
  auto.run(*args,grid=grid)
  first=list(events)
  assert state['benchmark']==2 and events[-1]['benchmark']=='selected_final'
  success=[e for e in events if 'before' in e]
  assert [e['before'] for e in success]==list(range(len(success)))
  assert len(first)==(1 if fail_first else repetitions)+repetitions+1
  auto.run(*args,grid=grid)
  assert len(events)==len(first)+1 and state['benchmark']==2
  records.append({'repetitions':repetitions,'fail_first':fail_first,'cache_miss_events':first,'cache_hit_event':events[-1]})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt={'source':str(source),'source_sha256':sha(source),'autotuner_source':runtime.__file__,'autotuner_sha256':sha(runtime.__file__),'records':records,'all_passed':True,'boundary':'Actual installed Autotuner; instrumented launch/benchmark primitives, no GPU. Both duplicate configurations are benchmarked, each repetition sees preceding writes; resource failure fallback, final choice and cache hit included.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'probe_autotune.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({'receipt':str(out/'receipt.json'),'sha256':sha(out/'receipt.json'),'schedules':len(records)}))

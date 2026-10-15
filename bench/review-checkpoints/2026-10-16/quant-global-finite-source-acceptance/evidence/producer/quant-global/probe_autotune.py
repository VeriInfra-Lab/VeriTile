from pathlib import Path
import ast,hashlib,importlib.util,json,sys
sys.dont_write_bytecode=True
root=Path(__file__).resolve().parent
out=root/'autotune-probes';out.mkdir(exist_ok=True)
source=Path('/mnt/VerifiableAILab/VerifiableAILab/zenan.li/VeriTile/bench/tritonbench_g/quantize_global/quantize_global.py')
prefix=source.read_text().split('################################################################')[0]
copied=out/'original_prefix.py';copied.write_text(prefix)
spec=importlib.util.spec_from_file_location('quantize_autotune_probe',copied)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
auto=module._quantize_global
import triton.runtime.autotuner as runtime
from triton.runtime.errors import OutOfResources
records=[]
class Tensor:
 def __init__(self,dtype):self.dtype=dtype
for repetitions in [1,2,5]:
 for fail_first in [False,True]:
  events=[];state={'version':0};calls=[]
  xs,scale,output=Tensor('float32'),Tensor('float32'),Tensor('int8')
  class Kernel:
   def run(self,*args,**kwargs):
    block=kwargs['BLOCK_SIZE']
    assert args==(xs,scale,output,2049)
    assert kwargs['grid']({'BLOCK_SIZE':block})==((2049+block-1)//block,)
    assert kwargs['num_warps']==4 and kwargs['num_stages']==(3 if block==1024 else 1)
    if fail_first and block==1024:
     events.append({'block':block,'failed':'OutOfResources','version':state['version']})
     raise OutOfResources(100,1,'test resource')
    previous=state['version'];state['version']+=1
    events.append({'block':block,'before':previous,'after':state['version']})
  def bench(call,quantiles):
   assert quantiles==(0.5,0.2,0.8)
   for _ in range(repetitions):call()
   # Deliberately select the last successful benchmark config.
   return [2 if events[-1]['block']==1024 else 1]*3
  auto.fn=Kernel();auto._do_bench=bench
  auto.__dict__.pop('do_bench',None);auto.cache={};auto.cache_results=False
  grid=lambda meta:((2049+meta['BLOCK_SIZE']-1)//meta['BLOCK_SIZE'],)
  auto.run(xs,scale,output,2049,grid=grid)
  first=list(events)
  assert events[-1]['block']==2048
  success=[e for e in events if 'before' in e]
  assert [e['before'] for e in success]==list(range(len(success)))
  expected_trials=(1 if fail_first else repetitions)+repetitions+1
  assert len(first)==expected_trials
  auto.run(xs,scale,output,2049,grid=grid)
  assert len(events)==len(first)+1 and events[-1]['block']==2048
  records.append({'repetitions':repetitions,'fail_first':fail_first,'cache_miss_events':first,'cache_hit_event':events[-1]})
receipt={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'autotuner_source':runtime.__file__,'autotuner_sha256':hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest(),'records':records,'all_passed':True,'boundary':'Actual installed Autotuner with instrumented kernel launch and benchmark primitives; tests scheduling, argument identity, repeated writes, swallowed resource failure, final choice and cache hit. No GPU calls.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(len(records),'actual Autotuner schedules passed')

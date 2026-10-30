"""Actual wrapper AST with ordinary API memory/identity adapters.
No Torch, CUDA or real allocator is executed. Injected exceptions check exact
call prefixes and partial adapter state, including a launch error after a store.
The complete actual kernel execution is covered separately by replay_candidate.
"""
import argparse, ast, hashlib, json, types
from pathlib import Path
P=Path(__file__).resolve().parents[1]
source=P/'iv_dependent_matmul_i64.py'
tree=ast.parse(source.read_text())
wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='iv_dependent_matmul_wrapper_i64')
wrapper.decorator_list=[]
code=compile(ast.Module(body=[wrapper],type_ignores=[]),str(source),'exec')
MODES=['pre_load','post_load','post_pre_mixed','post_load_two_iters','post_load_three_iters']
STAGES=['device','randA','randB','emptyC','launch']

class InjectedFailure(RuntimeError): pass
class Tensor:
    def __init__(self,shape,device):
        self.shape=shape; self.device=device; self.dtype='float32'
        self.cells=[None]*(shape[0]*shape[1])
    def stride(self,axis): return [max(self.shape[1],1),1][axis]
class API:
    float32="float32"
    def get_default_dtype(self): return self.float32
    def __init__(self,fail):
        self.fail=fail;self.events=[];self.tensors=[];self.allocated=[]
        self.caller=['existing buffer cell 0','existing buffer cell 1']
        self.before=self.caller.copy()
    def snapshot(self):
        return {'caller':self.caller.copy(),'allocations':[t.cells.copy() for t in self.allocated]}
    def call(self,name):
        self.events.append(name)
        if self.fail==name:
            self.error_state=self.snapshot()
            raise InjectedFailure(name)
    def device(self,device): self.call('device');return types.SimpleNamespace(type='cuda',name=device)
    def rand(self,shape,device):
        self.call('randA' if len(self.tensors)==0 else 'randB')
        t=Tensor(shape,device);t.cells[:]=[len(self.tensors)+1]*len(t.cells)
        self.tensors.append(t);self.allocated.append(t);return t
    def empty(self,shape,device):
        self.call('emptyC');self.output=Tensor(shape,device)
        self.allocated.append(self.output);return self.output
    def __getitem__(self,grid):
        def launch(a,b,c,M,N,K,*strides,**meta):
            self.call('launch')
            assert a is self.tensors[0] and b is self.tensors[1] and c is self.output
            assert a.shape==(M,K) and b.shape==(K,N) and c.shape==(M,N)
            assert strides==(max(K,1),1,max(N,1),1,max(N,1),1)
            assert grid(meta)==(((M+31)//32)*((N+31)//32),)
            assert meta['num_stages']==(4 if meta['type']=='post_load_three_iters' else 3)
            if self.fail=='launch_partial':
                if c.cells: c.cells[0]='partial launch store'
                self.error_state=self.snapshot()
                raise InjectedFailure('launch_partial')
        return launch
records=[]
for mode in MODES:
 for M,K,N in [(35,65,37),(1,0,3),(0,32,37),(35,32,0)]:
  for fail in [None]+STAGES+['launch_partial']:
    api=API(fail)
    env={'torch':api,'triton':types.SimpleNamespace(cdiv=lambda n,b:(n+b-1)//b),'iv_dependent_matmul_kernel_i64':api}
    exec(code,env)
    try:
        result=env['iv_dependent_matmul_wrapper_i64'](M,K,N,32,32,32,type=mode,device='cuda')
    except InjectedFailure as e:
        assert fail is not None and str(e)==fail
        failed_stage='launch' if fail=='launch_partial' else fail
        assert api.events==STAGES[:STAGES.index(failed_stage)+1]
        assert api.snapshot()==api.error_state
        assert len(api.allocated)=={'device':0,'randA':0,'randB':1,'emptyC':2,'launch':3}[failed_stage]
        if fail=='launch_partial' and M*N: assert api.output.cells[0]=='partial launch store'
        outcome='exception_and_exact_partial_state_propagated'
    else:
        assert (fail is None or (M*N==0 and fail in ("launch","launch_partial")))
        assert api.events==(STAGES if M*N else STAGES[:-1])
        assert result is api.output
        outcome='same_output_object_returned'
    assert api.caller==api.before
    for i,t in enumerate(api.tensors): assert t.cells==[i+1]*len(t.cells)
    records.append({'selector':mode,'shape':[M,K,N],'failure_at':fail,'events':api.events,'outcome':outcome,
      'partial_state_sha256':hashlib.sha256(json.dumps(api.snapshot(),sort_keys=True).encode()).hexdigest(),
      'allocated_extents':[len(t.cells) for t in api.allocated],
      'partial_store_observed':fail=='launch_partial' and M*N>0,
      'caller_preserved_by_adapter':True,'input_snapshots_preserved':True})
result={'schema':1,'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'method':'actual wrapper AST with ordinary external API memory contract adapters and injected failures, including partial launch effects',
 'native_gpu_execution':False,'real_torch_allocator_execution':False,'records':records,
 'checks':['device conversion precedes A random allocation','A precedes B random allocation','B precedes output allocation','output allocation precedes launch','launch precedes successful return','each injected exception propagates immediately','exact partial memory propagates on errors','ordinary allocation adapters preserve caller memory and earlier input snapshots','post-submission launch failure may expose partial output changes','successful return preserves output object identity','five documented selectors and actual num_stages retained']}
a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'cases':len(records),'successes':sum(r['outcome']=='same_output_object_returned' for r in records),'failure_prefixes':sum(r['outcome']!='same_output_object_returned' for r in records),'gpu':False}))

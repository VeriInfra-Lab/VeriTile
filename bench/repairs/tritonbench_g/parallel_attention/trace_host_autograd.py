"""Record actual ATen host/autograd instruction DAG, without native GPU claims."""
from pathlib import Path
import json,hashlib
import torch
from torch.utils._python_dispatch import TorchDispatchMode
D=Path(__file__).resolve().parent
class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.names={};self.keep=[];self.ops=[];self.phase='forward'
    def name(self,t):
        if id(t) not in self.names:
            self.names[id(t)]=f't{len(self.names)}';self.keep.append(t)
        return self.names[id(t)]
    def rep(self,x):
        if isinstance(x,torch.Tensor):return {'tensor':self.name(x),'shape':list(x.shape),'dtype':str(x.dtype),'stride':list(x.stride())}
        if isinstance(x,(list,tuple)):return [self.rep(v) for v in x]
        if isinstance(x,dict):return {k:self.rep(v) for k,v in x.items()}
        return x
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};inputs=self.rep(args);keys=self.rep(kwargs);out=func(*args,**kwargs)
        self.ops.append({'phase':self.phase,'operator':str(func),'args':inputs,'kwargs':keys,'result':self.rep(out)})
        return out
trace=Trace()
o=torch.empty(1,2,3,5,requires_grad=True);z=torch.empty(1,2,3,requires_grad=True);dy=torch.empty_like(o)
trace.names.update({id(o):'O',id(z):'Z',id(dy):'dY'});trace.keep.extend([o,z,dy])
with trace:
    y=o/(z[...,None]+.01)
    trace.phase='backward'
    do,dz=torch.autograd.grad(y,(o,z),dy)
record={'torch_version':torch.__version__,'torch_git_version':torch.version.git_version,'device':'cpu','execution_scope':'actual ATen dispatcher trace of unchanged wrapper division and its autograd backward; no CUDA kernel numerical claim','trace':trace.ops,'outputs':{'do':trace.name(do),'dz':trace.name(dz)},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'GPU_execution':False}
(D/'host-autograd-trace.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'forward':[x['operator'] for x in trace.ops if x['phase']=='forward'],'backward':[x['operator'] for x in trace.ops if x['phase']=='backward'],'outputs':record['outputs']}))

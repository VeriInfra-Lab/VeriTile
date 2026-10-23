"""Actual CPU JIT/autograd controls for the checked wrapper and resident body."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ceops_support',D.parent/'cross_entropy1/Proofs/probe.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
torch=support.torch;torch.set_num_threads(1)
SOURCE=D/'cross_entropy_ops_repaired.py';BODY=D/'cross_entropy_ops_conditional.py'
assert hashlib.sha256(BODY.read_bytes()).hexdigest()=='b690a712bd968686ee78818e81dbbb21a8d15331e132eb1c994d1ed3eb2b4ba5'
# The installed-language key continues to name the exact body that is called.
sourceAST=(D/'Proofs/SourceAST.lean').read_text()
assert hashlib.sha256(BODY.read_bytes()).hexdigest() in sourceAST
assert ast.dump(ast.parse(BODY.read_text()),include_attributes=False) == ast.literal_eval(sourceAST.split('def pythonAST : String := ',1)[1].split('\n\ndef pythonDigest',1)[0])
class Runner:
    def __init__(self,name):
        self.allocations=[];self.launches=[];self.launch_tensors=[]
        spec=importlib.util.spec_from_file_location('checked_ceops_'+name,SOURCE);self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        host=support.TorchHost(self);self.module.torch=host;self.module._source.torch=host
        for n in ('cross_entropy_fwd_kernel','cross_entropy_bwd_kernel'):
            original=getattr(self.module._source,n)
            assert getattr(self.module,n) is original
            setattr(self.module._source,n,support.Kernel(self,original,n))
        f=self.module.CrossEntropyLoss.forward;b=self.module.CrossEntropyLoss.backward
        def forward(ctx,*args):
            out=f(ctx,*args);self.saved=ctx.to_save;return out
        def backward(ctx,g,gz):
            self.upstream=g;self.backward_start=len(self.allocations);out=b(ctx,g,gz);self.backward_result=out;return out
        self.module.CrossEntropyLoss.forward=staticmethod(forward);self.module.CrossEntropyLoss.backward=staticmethod(backward)
def reference(x,y,g,s,a,z,ignored):
    u=x.double()*a;l=torch.logsumexp(u,1);safe=y.clone();safe[y==ignored]=0
    zz=z*l*l;loss=l-(1-s)*u.gather(1,safe[:,None]).flatten()-s*u.sum(1)/x.shape[1]+zz
    grad=(1+2*z*l[:,None])*torch.softmax(u,1)-s/x.shape[1]
    grad.scatter_add_(1,safe[:,None],torch.full((len(y),1),-(1-s),dtype=torch.float64));grad*=(g.double()*a)[:,None]
    active=y!=ignored
    return torch.where(active,loss,0),torch.where(active,zz,0),torch.where(active[:,None],grad,0)
dense=lambda:torch.tensor([[0.,1.,2.],[2.,0.,1.]])
labels=lambda:torch.tensor([0,1])
r=Runner('guards');m=r.module;rejected=[];cases=[]
def reject(name,thunk,kind=(ValueError,TypeError)):
    try:thunk()
    except kind:rejected.append(name)
    else:raise AssertionError('admitted '+name)
def check(**kw):
    k=dict(x=dense(),y=labels(),s=0.,a=1.,z=0.,ignored=-100,inplace=False,group=None);k.update(kw)
    return m._check_forward(*[k[n] for n in ('x','y','s','a','z','ignored','inplace','group')])
reject('CPU',lambda:check())
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)):
    for name,k in [
        ('non Tensor',{'x':[]}),('FP64',{'x':dense().double()}),('int32 labels',{'y':labels().int()}),
        ('rank1',{'x':torch.arange(3.)}),('label shape',{'y':labels()[:1]}),('empty classes',{'x':torch.empty(2,0)}),
        ('column stride',{'x':torch.arange(12.).reshape(2,6)[:,::2]}),('label stride',{'y':torch.tensor([0,9,1])[::2]}),
        ('negative logits',{'x':torch._neg_view(dense())}),('negative labels',{'y':torch._neg_view(labels())}),
        ('integer smoothing',{'s':0}),('integer z',{'z':0}),('integer logit scale',{'a':1}),
        ('nan smoothing',{'s':float('nan')}),('low smoothing',{'s':-.1}),('large smoothing',{'s':1.1}),
        ('nan z',{'z':float('nan')}),('negative z',{'z':-.1}),('infinite z',{'z':float('inf')}),
        ('nan logit scale',{'a':float('nan')}),('infinite logit scale',{'a':float('inf')}),
        ('bool ignored',{'ignored':True}),('small ignored',{'ignored':-(1<<63)-1}),('large ignored',{'ignored':1<<63}),
        ('integer inplace',{'inplace':1}),('process group',{'group':object()}),
        ('negative target',{'y':torch.tensor([-1,1])}),('large target',{'y':torch.tensor([0,3])})]:
        reject(name,lambda k=k:check(**k))
    for name,g in [('upstream object',[]),('upstream dtype',torch.ones(2,dtype=torch.float64)),
                   ('upstream shape',torch.ones(3)),('negative upstream',torch._neg_view(torch.ones(2)))]:
        reject(name,lambda g=g:m._check_backward(g,dense()))
    def case(name,x,y,s=0.,a=1.,z=0.,ignored=-100,inplace=False,g=None,summed=False,exceptional=False):
        if g is None:g=torch.ones(len(y))
        expected=None if exceptional else reference(x,y,g,s,a,z,ignored)
        backing={t.untyped_storage().data_ptr():(t,support.storage_bytes(t)) for t in (x,y,g)}
        old_labels=y.clone();run=Runner(name);x.requires_grad_(True)
        loss,zloss=run.module.cross_entropy_loss(x,y,s,a,z,ignored,inplace)
        assert not zloss.requires_grad and zloss.grad_fn is None
        grad,=torch.autograd.grad(loss.sum(),x) if summed else torch.autograd.grad(loss,x,g)
        returned=run.backward_result[0]
        assert len(run.backward_result)==8 and all(v is None for v in run.backward_result[1:])
        assert torch.equal(run.saved[2],old_labels)
        assert returned is (run.saved[3] if inplace else run.allocations[run.backward_start])
        if inplace:assert returned is x
        if exceptional:
            active=old_labels==ignored
            for out in (loss,zloss,grad):assert torch.equal(out[active],torch.zeros_like(out[active]))
        else:
            for out,want in zip((loss,zloss,grad),expected):torch.testing.assert_close(out.double(),want,atol=8e-5,rtol=8e-5)
        allowed=support.footprint(returned) if inplace else set()
        for base,(t,b) in backing.items():
            changed=torch.nonzero(support.storage_bytes(t)!=b).flatten().tolist()
            assert all(base+i in allowed for i in changed)
        if len(y):
            assert run.saved[2].data_ptr()!=y.data_ptr()
            assert run.launch_tensors[-1]['dloss_ptr'].data_ptr()!=run.upstream.data_ptr()
        cases.append(name);print(name,'passed',flush=True)
    case('dense',dense(),labels())
    case('broadcast_sum_upstream',dense(),labels(),s=.1,summed=True)
    for a in (1.,0.,-2.):case('logit_scale_'+str(a),dense(),labels(),s=.2,a=a,z=.1,g=torch.tensor([1.,2.]))
    case('mixed_ignored',dense(),torch.tensor([-100,1]),s=.1,z=.2)
    case('int64_min_ignored',dense(),torch.tensor([-(1<<63),1]),ignored=-(1<<63))
    case('offset_inplace',torch.arange(12.).reshape(4,3)[1:3],labels(),inplace=True)
    case('strided_upstream',dense(),labels(),g=torch.tensor([1.,9.,2.])[::2])
    x=dense();case('upstream_alias_inplace',x,labels(),g=x[:,0],inplace=True)
    y=labels();case('label_alias_inplace',y.view(torch.float32).reshape(2,2),y,inplace=True)
    case('one_class',torch.tensor([[2.],[1.]]),torch.tensor([0,0]),z=.2)
    case('empty_batch',torch.empty(0,3),torch.empty(0,dtype=torch.int64),g=torch.empty(0))
    for tail in (1,2,3):case('partial_byte_tail_'+str(tail),support.partial_float_storage(tail),labels(),inplace=tail==2)
    case('ignored_nonfinite',torch.tensor([[float('nan'),1.],[float('inf'),0.],[0.,float('-inf')]]),
         torch.full((3,),-100,dtype=torch.int64),z=.2,exceptional=True)
    case('backward_tail_4097',torch.zeros(1,4097),torch.tensor([4096]),s=.1,a=-2.,z=.2)
    case('local_split_65537',torch.zeros(1,65537),torch.tensor([65536]),s=.1,a=-2.,z=.2)
    case('local_split_zero_z',torch.zeros(1,65537),torch.tensor([-100]),a=0.)
    version=Runner('version');x=dense().requires_grad_();loss,z=version.module.cross_entropy_loss(x,labels())
    with torch.no_grad():x.add_(1)
    reject('saved version',lambda:torch.autograd.grad(loss.sum(),x),RuntimeError)
    assert len(version.launches)==1
assert not r.launches
print(f'{len(rejected)} rejected; {len(cases)} autograd/source/frame cases passed; no GPU conformance claim')

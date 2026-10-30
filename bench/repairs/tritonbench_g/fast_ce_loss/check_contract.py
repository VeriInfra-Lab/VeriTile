"""CPU interpreter/autograd checks of the exact checked entry and bound body."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.dont_write_bytecode=True
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fastce_support',D.parent/'cross_entropy1/Proofs/probe.py')
support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
torch=support.torch;torch.set_num_threads(1)
import numpy as np
import triton.language as tl
from triton.runtime.interpreter import TensorHandle
def cpu_tanh(value):
    # This Triton interpreter lacks libdevice.tanh. Only that scalar port is
    # supplied by NumPy; no native libdevice numerical conformance is claimed.
    return tl.tensor(TensorHandle(np.tanh(value.handle.data),value.dtype.scalar),value.type)
SOURCE=D/'fast_ce_loss_repaired.py';BODY=D/'fast_ce_loss_body.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() in (D/'Contract.lean').read_text()
sourceAST=(D/'Proofs/SourceAST.lean').read_text()
assert hashlib.sha256(BODY.read_bytes()).hexdigest() in sourceAST
assert ast.dump(ast.parse(BODY.read_text()),include_attributes=False)==ast.literal_eval(sourceAST.split('def pythonAST : String := ',1)[1].split('\n\ndef pythonDigest',1)[0])
conditional=D/'fast_ce_loss_conditional.py'
assert hashlib.sha256(conditional.read_bytes()).hexdigest()=='032f2018d4b664214dd1e147bcfd8b5956a0bb75a9018f1a9b25fba8484d6f90'
old=conditional.read_text().split('def test_fast_cross_entropy_loss_with_backward',1)[0]
fixed=BODY.read_text().replace('tl.where(label_idx == -100, 0.0, dloss * y)','dloss * y')
old_ast=ast.parse(old)
assert ast.unparse(old_ast.body.pop())=='import torch'  # benchmark-harness import
assert ast.dump(old_ast,include_attributes=False)==ast.dump(ast.parse(fixed),include_attributes=False)
class Runner:
    def __init__(self,name,path=SOURCE):
        self.allocations=[];self.launches=[];self.launch_tensors=[]
        spec=importlib.util.spec_from_file_location('checked_fastce_'+name,path);self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.body=getattr(self.module,'_source',self.module)
        self.body.torch=support.TorchHost(self)
        self.body.triton_tanh=cpu_tanh
        for n in ('_cross_entropy_forward','_chunked_cross_entropy_forward','_cross_entropy_backward'):
            setattr(self.body,n,support.Kernel(self,getattr(self.body,n),n))
        f=self.body.Fast_CrossEntropyLoss.forward;b=self.body.Fast_CrossEntropyLoss.backward
        def forward(ctx,*args):
            out=f(ctx,*args);self.saved=ctx.to_save;return out
        def backward(ctx,g):
            self.upstream=g;out=b(ctx,g);self.backward_result=out;return out
        self.body.Fast_CrossEntropyLoss.forward=staticmethod(forward);self.body.Fast_CrossEntropyLoss.backward=staticmethod(backward)
def reference(x,y,cap,scale,items,g):
    v=x.detach().double();s=scale if scale!=0 else 1;v=v*s
    partial=torch.tanh(v/cap) if cap!=0 else None
    v=cap*partial if cap!=0 else v
    flat=v.reshape(-1,v.shape[-1]);target=y.flatten();safe=target.clone();safe[target==-100]=0
    active=target!=-100;denom=int(active.sum()) if items is None else items
    losses=torch.logsumexp(flat,1)-flat.gather(1,safe[:,None]).flatten()
    loss=torch.where(active,losses,0).sum()/denom
    grad=torch.softmax(flat,1);grad.scatter_add_(1,safe[:,None],-torch.ones((len(target),1),dtype=torch.float64))
    grad=grad.reshape(v.shape)*s
    if cap!=0:grad*=1-partial*partial
    grad*=g.double()/denom
    grad=torch.where((y!=-100)[...,None],grad,0)
    return loss,grad
x0=lambda:torch.tensor([[[0.,1.,2.],[2.,0.,1.]]])
y0=lambda:torch.tensor([[0,1]])
r=Runner('guards');m=r.module;rejected=[];cases=[]
def reject(name,thunk,kind=(ValueError,TypeError)):
    try:thunk()
    except kind:rejected.append(name)
    else:raise AssertionError('admitted '+name)
def check(**kw):
    k=dict(x=x0(),y=y0(),cap=0,scale=0,items=None);k.update(kw)
    return m._check_forward(*[k[n] for n in ('x','y','cap','scale','items')])
reject('CPU',lambda:check())
with patch.object(torch.Tensor,'is_cuda',property(lambda _:True)),patch.object(torch.cuda,'current_device',lambda:None):
    for name,k in [
        ('object',{'x':[]}),('FP16',{'x':x0().half()}),('FP64',{'x':x0().double()}),('int32 label',{'y':y0().int()}),
        ('sparse logits',{'x':x0().to_sparse()}),('device mismatch',{'x':torch.empty(1,2,3,device='meta')}),
        ('rank2',{'x':x0().view(2,3)}),('label shape',{'y':y0().flatten()}),('zero classes',{'x':torch.empty(1,2,0)}),
        ('logit stride',{'x':torch.arange(12.).reshape(1,2,6)[...,::2]}),('label stride',{'y':torch.tensor([[0,9,1]])[:,::2]}),
        ('negative logit',{'x':torch._neg_view(x0())}),('negative label',{'y':torch._neg_view(y0())}),
        ('bool cap',{'cap':False}),('Tensor cap',{'cap':torch.tensor(1.)}),('large cap int',{'cap':1<<63}),
        ('small cap int',{'cap':-(1<<63)-1}),('nan cap',{'cap':float('nan')}),('inf cap',{'cap':float('inf')}),
        ('bool scale',{'scale':True}),('Tensor scale',{'scale':torch.tensor(1.)}),('inf scale',{'scale':float('-inf')}),
        ('bool items',{'items':True}),('Tensor items',{'items':torch.tensor(2.)}),('zero items',{'items':0}),
        ('negative items',{'items':-1}),('nan items',{'items':float('nan')}),('large items',{'items':1<<63}),
        ('all ignored default',{'y':torch.full((1,2),-100)}),('negative label value',{'y':torch.tensor([[-1,1]])}),
        ('large label value',{'y':torch.tensor([[0,3]])})]:
        reject(name,lambda k=k:check(**k))
    y=y0();reject('label allocation alias',lambda:check(x=y.view(torch.float32).reshape(1,2,2),y=y))
    reject('empty default',lambda:check(x=torch.empty(0,2,3),y=torch.empty(0,2,dtype=torch.int64)))
    reject('row grid',lambda:check(x=torch.empty((1<<31),1,1,device='meta'),
                                 y=torch.empty((1<<31),1,device='meta',dtype=torch.int64)))
    reject('column grid',lambda:check(x=torch.empty(1,1,4096*65535+1,device='meta'),
                                    y=torch.empty(1,1,device='meta',dtype=torch.int64)))
    reject('unaligned logits',lambda:check(x=torch.frombuffer(bytearray(25),dtype=torch.float32,offset=1,count=6).reshape(1,2,3)))
    reject('unaligned labels',lambda:check(y=torch.frombuffer(bytearray(17),dtype=torch.int64,offset=1,count=2).reshape(1,2)))
    with patch.object(torch.cuda,'current_device',lambda:1):reject('current device',lambda:check())
    for name,g in [('upstream object',[]),('upstream dtype',torch.tensor(1.,dtype=torch.float64)),
                   ('upstream vector',torch.ones(1)),('upstream negative bit',torch._neg_view(torch.tensor(1.)))]:
        reject(name,lambda g=g:m._check_upstream(g,x0(),y0()))
    y=y0();reject('upstream labels alias',lambda:m._check_upstream(y.view(torch.float32).flatten()[0],x0(),y))
    def case(name,x,y,cap=0,scale=0,items=None,g=None,exceptional=False):
        if g is None:g=torch.tensor(1.)
        expected=None if exceptional else reference(x,y,cap,scale,items,g)
        backing={t.untyped_storage()._cdata:(t,support.storage_bytes(t)) for t in (x,y,g)}
        old_labels=y.clone();run=Runner(name);x.requires_grad_(True)
        loss=run.module.fast_cross_entropy_loss(x,y,cap,scale,items)
        grad,=torch.autograd.grad(loss,x,g)
        returned=run.backward_result[0]
        assert len(run.backward_result)==4 and all(v is None for v in run.backward_result[1:])
        assert returned is run.saved[0]
        assert returned.untyped_storage()._cdata==x.untyped_storage()._cdata
        assert returned.storage_offset()==x.storage_offset()
        assert grad.untyped_storage()._cdata==x.untyped_storage()._cdata and grad.storage_offset()==x.storage_offset()
        assert grad.shape==x.shape and loss.shape==()
        assert torch.equal(y,old_labels)
        if exceptional:
            assert loss.item()==0 and torch.equal(grad,torch.zeros_like(grad))
            assert torch.equal(grad.view(torch.int32),torch.zeros_like(grad.view(torch.int32)))
        else:
            for out,want in zip((loss,grad),expected):torch.testing.assert_close(out.double(),want,atol=9e-5,rtol=9e-5)
        allowed=support.footprint(x)
        for t,b in backing.values():
            base=t.untyped_storage().data_ptr();changed=torch.nonzero(support.storage_bytes(t)!=b).flatten().tolist()
            assert all(base+i in allowed for i in changed)
        assert run.upstream.stride()==(0,) or y.numel()<=1
        cases.append(name);print(name,'passed',flush=True)
    case('defaults',x0(),y0())
    case('float_signed_zero',x0(),y0(),cap=-0.,scale=-0.)
    for cap in (0,.5,-.5):
        for scale in (0,1.5,-2.):case('cap_scale_'+str((cap,scale)),x0(),y0(),cap=cap,scale=scale,items=3.5,g=torch.tensor(-2.))
    case('mixed_ignored',x0(),torch.tensor([[-100,1]]))
    case('explicit_all_ignored',x0(),torch.full((1,2),-100),items=2)
    case('one_class',torch.tensor([[[2.],[1.]]]),torch.tensor([[0,0]]),cap=-1.,scale=-2.)
    case('empty_batch',torch.empty(0,2,3),torch.empty(0,2,dtype=torch.int64),items=2.)
    case('empty_sequence',torch.empty(1,0,3),torch.empty(1,0,dtype=torch.int64),items=2)
    case('offset_storage',torch.arange(18.).reshape(3,2,3)[1:2],torch.tensor([[9,9],[0,1],[9,9]])[1:2])
    for tail in (1,2,3):case('partial_byte_tail_'+str(tail),support.partial_float_storage(tail).view(1,2,3),y0())
    x=x0();case('upstream_logit_alias',x,y0(),g=x.flatten()[1])
    case('ignored_nonfinite',torch.tensor([[[float('nan'),1.],[float('inf'),0.],[0.,float('-inf')]]]),torch.full((1,3),-100),items=1,exceptional=True)
    case('ignored_nonfinite_cap_scale',torch.tensor([[[float('nan'),1.],[float('inf'),0.]]]),torch.full((1,2),-100),cap=-1.,scale=-2.,items=1,g=torch.tensor(float('nan')),exceptional=True)
    case('backward_tail4097',torch.zeros(1,1,4097),torch.tensor([[4096]]),cap=-1.,scale=-2.)
    case('forward_split65537',torch.zeros(1,1,65537),torch.tensor([[65536]]),cap=-1.,scale=-2.)
    hook=Runner('hook_alias');x=x0().requires_grad_();y=y0();before=support.storage_bytes(x)
    loss=hook.module.fast_cross_entropy_loss(x,y)
    reject('actual hook label alias',lambda:torch.autograd.grad(loss,x,y.view(torch.float32).flatten()[0]))
    assert len(hook.launches)==1 and torch.equal(before,support.storage_bytes(x))
    hook=Runner('hook_current');x=x0().requires_grad_();loss=hook.module.fast_cross_entropy_loss(x,y0())
    with patch.object(torch.cuda,'current_device',lambda:1):
        reject('actual hook current device',lambda:torch.autograd.grad(loss,x))
    assert len(hook.launches)==1
    version=Runner('version');x=x0().requires_grad_();loss=version.module.fast_cross_entropy_loss(x,y0())
    with torch.no_grad():x.add_(1)
    reject('saved version',lambda:torch.autograd.grad(loss,x),RuntimeError)
    assert len(version.launches)==1
assert not r.launches
# Separate, reproducible original-source counterexample: IEEE zero times NaN.
original=D.parents[2]/'tritonbench_g/fast_ce_loss/fast_ce_loss.py'
original_body=original.read_text().split('def test_fast_cross_entropy_loss_with_backward',1)[0]
with tempfile.TemporaryDirectory(prefix='veritile-fastce-original-') as temp:
    path=Path(temp)/'original_body.py';path.write_text(original_body)
    run=Runner('original_ignored_nan',path)
    class OriginalHost(support.TorchHost):
        def empty(self,*args,**kwargs):
            # The original hardcodes cuda:0 for outputs; adapt allocation only.
            assert kwargs['device']=='cuda:0'
            kwargs['device']='cpu'
            return super().empty(*args,**kwargs)
    run.body.torch=OriginalHost(run)
    x=torch.tensor([[[float('nan')]]],requires_grad=True);y=torch.tensor([[-100]])
    loss=run.module.fast_cross_entropy_loss(x,y,n_items=1)
    grad,=torch.autograd.grad(loss,x)
    assert loss.item()==0 and torch.isnan(grad).all()
    print('original ignored-NaN counterexample reproduced',flush=True)
print(f'{len(rejected)} rejected; {len(cases)} checked autograd/source/frame cases passed; no GPU conformance claim')

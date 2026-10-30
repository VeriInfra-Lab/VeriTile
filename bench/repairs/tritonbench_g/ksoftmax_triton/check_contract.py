"""Actual Python source with CPU backing storage; no native numerical claim."""
import ast
import importlib.util
import itertools
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
spec=importlib.util.spec_from_file_location('masked_cpu_helpers',HERE.parent/'masked_add_cuda/check_contract.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
Array=helpers.Array

class Pointer:
    def __init__(self,tensor,offsets):self.tensor,self.offsets=tensor,np.asarray(offsets)
    def __add__(self,offset):return Pointer(self.tensor,self.offsets+offset)
class Tensor(helpers.Tensor):
    def __add__(self,offsets):return Pointer(self,offsets)
class Language(helpers.Language):
    int64=np.int64
    float32=np.float32
    def program_id(self,axis):return np.asarray(self.pids[axis],dtype=np.int32).view(Array)
    def arange(self,start,stop):
        assert stop>0 and not stop&(stop-1)
        return np.arange(start,stop,dtype=np.int32).view(Array)
    def load(self,pointer,mask,other=0):
        tensor,offsets=pointer.tensor,pointer.offsets
        raw=tensor.raw();active=offsets[np.asarray(mask)]
        if np.any(active<0) or np.any(active>=len(raw)):raise IndexError('source load out of bounds')
        result=np.full(offsets.shape,other,dtype=raw.dtype).view(Array)
        result[np.asarray(mask)]=raw[active]
        return result
    def store(self,pointer,value,mask):
        return super().store((pointer.tensor,pointer.offsets),value,mask)
    def where(self,c,x,y):return np.asarray(np.where(c,x,y)).view(Array)
    def max(self,x,axis=0):return np.asarray(np.max(x,axis=axis)).view(Array)
    def sum(self,x,axis=0):return np.asarray(np.sum(x,axis=axis)).view(Array)
    def exp(self,x):return np.asarray(np.exp(x)).view(Array)
    def log(self,x):return np.asarray(np.log(x)).view(Array)
class JIT:
    def __init__(self,f,tl,calls):self.f,self.tl,self.calls=f,tl,calls
    def __getitem__(self,grid):
        def launch(*args,**kw):
            fusion=kw.pop('enable_fp_fusion',None)
            K=args[-1];D=1<<(max(1,K)-1).bit_length()
            kw.update(DEPTH=D,IS_FP16=False)
            self.calls.append((self.f.__name__,grid,K,D,fusion))
            for b,r in itertools.product(range(grid[0]),range(grid[1])):
                self.tl.pids=(b,r)
                with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                    self.f(*args,**kw)
        return launch

def load(path):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and not n.name.startswith('test_')]
    # Python's float literal .to is a Triton scalar cast; it has no effect in
    # this FP32 CPU projection. Integer widening remains interpreted by Array.
    class Literals(ast.NodeTransformer):
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and isinstance(n.func.value,ast.Name) and n.func.value.id in ('off','zero'):
                return ast.copy_location(n.func.value,n)
            return n
    module=ast.fix_missing_locations(Literals().visit(ast.Module(body=nodes,type_ignores=[])))
    tl,calls=Language(),[]
    tri=SimpleNamespace(next_power_of_2=lambda n:1<<(max(n,1)-1).bit_length(),Config=lambda *a,**k:None)
    tri.jit=lambda f:JIT(f,tl,calls)
    tri.autotune=lambda **k:lambda f:f
    tri.heuristics=lambda *a,**k:lambda f:f
    @contextmanager
    def device(d):
        assert d=='cuda:0'
        yield
    tor=SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,float16=torch.float16,
        cuda=SimpleNamespace(device=device),empty=lambda *a,**k:Tensor(torch.empty(*a)))
    ns={'torch':tor,'triton':tri,'tl':tl}
    exec(compile(module,str(path),'exec'),ns)
    return ns,calls

def source_binding():
    original=ast.parse((ROOT/'bench/tritonbench_g/ksoftmax_triton/ksoftmax_triton.py').read_text())
    candidate=ast.parse((HERE/'ksoftmax_triton_repaired.py').read_text())
    class Natural(ast.NodeTransformer):
        def visit_arg(self,n):n.annotation=None;return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and n.func.attr=='to' and len(n.args)==1 and ast.unparse(n.args[0])=='tl.int64':return n.func.value
            return n
    for name in ('_softmax','_softmax_backward'):
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in candidate.body if isinstance(n,ast.FunctionDef) and n.name==name)
        if name=='_softmax_backward':
            fixed=[n for n in b.body if isinstance(n,ast.If) and ast.unparse(n.test)=='CAUSAL' and len(n.body)==1 and ast.unparse(n.body[0]).startswith('grad_in = tl.where')]
            assert len(fixed)==1 and ast.unparse(fixed[0].body[0])=='grad_in = tl.where(k <= n, grad_in, 0.0)'
            b.body.remove(fixed[0])
        assert ast.dump(Natural().visit(a))==ast.dump(Natural().visit(b)),name

def tensor(values):return Tensor(torch.as_tensor(values,dtype=torch.float32).clone())
def forward_reference(x,m,mode,log,causal):
    score=x.value.clone()
    if mode is not None:score+=m.value[None,:,:] if mode=='qk' else m.value[:,None,:]
    if causal:score.masked_fill_(torch.arange(score.shape[2])[None,:]>torch.arange(score.shape[1])[:,None],-float('inf'))
    return torch.log_softmax(score,-1) if log else torch.softmax(score,-1)
def backward_reference(g,o,log,causal):
    gv,ov=g.value.clone(),o.value.clone()
    active=torch.arange(ov.shape[2])[None,:]<=torch.arange(ov.shape[1])[:,None]
    if causal:gv.masked_fill_(~active,0);ov.masked_fill_(~active,0)
    expected=gv-ov.exp()*gv.sum(-1,keepdim=True) if log else ov*(gv-(gv*ov).sum(-1,keepdim=True))
    if causal:expected.masked_fill_(~active,0)
    return expected

def main():
    source_binding()
    repaired,calls=load(HERE/'ksoftmax_triton_repaired.py')
    original,_=load(ROOT/'bench/tritonbench_g/ksoftmax_triton/ksoftmax_triton.py')
    accepted=rejected=failures=0
    for K,mode,log,causal in itertools.product((1,3,8,33),(None,'qk','bk'),(False,True),(False,True)):
        x=tensor(torch.linspace(-1,1,6*K).reshape(2,3,K));y=tensor(torch.zeros_like(x.value))
        m=None if mode is None else tensor(torch.linspace(-.5,.5,(3 if mode=='qk' else 2)*K).reshape(-1,K))
        expected=forward_reference(x,m,mode,log,causal)
        assert repaired['softmax'](y,x,m,log,mode,causal) is None
        torch.testing.assert_close(y.value,expected,atol=3e-6,rtol=3e-6)
        assert calls[-1][-1] is True
        accepted+=1
    for K,log,causal in itertools.product((1,3,8,33),(False,True),(False,True)):
        x=tensor(torch.linspace(-1,1,6*K).reshape(2,3,K));o=tensor(forward_reference(x,None,None,log,causal))
        g=tensor(torch.linspace(-1,2,6*K).reshape(2,3,K));d=tensor(torch.zeros_like(g.value))
        expected=backward_reference(g,o,log,causal)
        assert repaired['softmax_backward'](d,g,o,log,causal) is None
        torch.testing.assert_close(d.value,expected,atol=3e-6,rtol=3e-6)
        accepted+=1
    for shape in ((0,2,3),(2,0,3),(2,3,0)):
        ts=[tensor(torch.empty(shape)) for _ in range(3)]
        for t in ts:t.metadata['forbid_pointer']=True
        old=len(calls)
        repaired['softmax'](*ts[:2],object())
        repaired['softmax_backward'](*ts)
        assert len(calls)==old and not any(t.pointer_queries for t in ts)
        accepted+=2
    # Shared read inputs, offset views, singleton-stride canonical addressing,
    # same-allocation disjoint spans, and full exterior storage frame.
    backing=torch.full((50,),77.)
    y=Tensor(backing[2:14].reshape(2,2,3));x=Tensor(backing[20:32].reshape(2,2,3));x.value.copy_(torch.arange(12).reshape(2,2,3))
    before=backing.clone();expected=forward_reference(x,None,None,False,False)
    repaired['softmax'](y,x);torch.testing.assert_close(y.value,expected)
    assert torch.equal(backing[:2],before[:2]) and torch.equal(backing[14:],before[14:]);accepted+=1
    x=tensor(torch.arange(3).reshape(1,1,3));y=tensor(torch.zeros(1,1,3));x.value=x.value.as_strided((1,1,3),(93,31,1))
    repaired['softmax'](y,x);torch.testing.assert_close(y.value,torch.softmax(x.value,-1));accepted+=1
    g=tensor(torch.tensor([[[2.,float('nan')]]]));o=tensor(torch.tensor([[[0.,-float('inf')]]]));d=tensor(torch.zeros(1,1,2))
    repaired['softmax_backward'](d,g,o,True,True);assert torch.equal(d.value,torch.zeros_like(d.value));accepted+=1
    g=tensor(torch.ones(1,2,3));d=tensor(torch.zeros(1,2,3));expected=backward_reference(g,g,False,False)
    repaired['softmax_backward'](d,g,g);torch.testing.assert_close(d.value,expected);accepted+=1
    def reject(fn,*args,**kw):
        nonlocal rejected
        before=len(calls)
        try:repaired[fn](*args,**kw)
        except (TypeError,ValueError):rejected+=1
        else:raise AssertionError('unsupported input accepted')
        assert len(calls)==before
    def args():return [tensor(torch.ones(1,2,3)) for _ in range(3)]
    for fn in ('softmax','softmax_backward'):
        for slot,meta in ((0,{'dtype':torch.float16}),(1,{'layout':torch.sparse_coo}),(0,{'is_cuda':False}),
                          (1,{'device':'cuda:1'}),(0,{'contiguous':False}),(1,{'negative':True}),
                          (0,{'shape':(1,1,3)})):
            ts=args();ts[slot].metadata.update(meta);reject(fn,*ts if fn.endswith('backward') else ts[:2])
        reject(fn,*args() if fn.endswith('backward') else args()[:2],log=1)
        reject(fn,*args() if fn.endswith('backward') else args()[:2],causal='yes')
        ts=args();ts[0]=None;reject(fn,*ts if fn.endswith('backward') else ts[:2])
        for slot in range(3 if fn.endswith('backward') else 2):
            ts=args();ts[slot].metadata['pointer']=ts[slot].value.data_ptr()+1;reject(fn,*ts if fn.endswith('backward') else ts[:2])
    for mode in ('bad','qk','bk'):
        reject('softmax',*args()[:2],None,mask_type=mode)
    reject('softmax',*args(),mask_type='qk')
    for shape in ((2**31,1,1),(1,2**31,1),(1,1,2**30+1)):
        ts=args()[:2]
        for t in ts:t.metadata.update(shape=shape,ndim=3)
        reject('softmax',*ts)
    ts=args()[:2]
    for t in ts:t.metadata['numel']=2**61
    reject('softmax',*ts)
    x=tensor(torch.zeros(1,2,3));m=tensor([[0.,0.,0.],[0.,5.,0.]]);y=tensor(torch.zeros_like(x.value))
    expected=forward_reference(x,m,'qk',False,False)
    original['softmax'](y,x,m,mask_type='qk');assert not torch.allclose(y.value,expected);failures+=1
    d=tensor(torch.zeros(1,1,2));g=tensor([[[2.,5.]]]);o=tensor([[[0.,-float('inf')]]])
    original['softmax_backward'](d,g,o,True,True);assert d.value[0,0,1]==-2;failures+=1
    raw=torch.tensor([0.,99.,1.,99.,2.]);x=Tensor(raw[::2].reshape(1,1,3));y=tensor(torch.zeros_like(x.value))
    reject('softmax',y,x);original['softmax'](y,x);assert not torch.allclose(y.value,torch.softmax(x.value,-1));failures+=1
    raw=torch.full((5,),77.);y=Tensor(raw[::2].reshape(1,1,3));x=tensor([[[0.,1.,2.]]])
    reject('softmax',y,x);original['softmax'](y,x);assert raw[1]!=77 and raw[4]==77;failures+=1
    x=tensor([[[0.,1.,2.]]]);x.value=torch._neg_view(x.value);y=tensor(torch.zeros_like(x.value))
    reject('softmax',y,x);original['softmax'](y,x);assert not torch.allclose(y.value,torch.softmax(x.value,-1));failures+=1
    raw=torch.tensor([0.,8.,0.,0.,0.,0.,0.]);x=Tensor(raw[:6].reshape(1,2,3));y=Tensor(raw[1:7].reshape(1,2,3))
    expected=forward_reference(x,None,None,False,False);reject('softmax',y,x);original['softmax'](y,x);assert not torch.allclose(y.value,expected);failures+=1
    ts=args();reject('softmax_backward',ts[1],ts[1],ts[2])
    print(f'passed: {accepted} admissions/branches, {rejected} rejections, {failures} original counterexamples')
    print('exact source arithmetic binding; CPU interpretation only; no native conformance claim')

if __name__=='__main__':main()

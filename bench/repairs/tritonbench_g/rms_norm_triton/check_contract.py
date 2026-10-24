"""Metadata/copy order and exact-source CPU controls; no GPU conformance claim."""
import ast
import contextlib
import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
os.environ['TRITON_INTERPRET']='1'
import torch
import triton
HERE=Path(__file__).resolve().parent
SOURCE=HERE/'rms_norm_triton_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rms_norm_triton/rms_norm_triton.py'

def node(path,name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
def expr(s):return ast.dump(ast.parse(s,mode='eval').body)
def source_checks():
    assert ast.dump(node(ORIGINAL,'rms_norm_kernel'))==ast.dump(node(SOURCE,'rms_norm_kernel'))
    fn=node(SOURCE,'forward')
    guards=[ast.dump(n.test) for n in fn.body if isinstance(n,ast.If)]
    expected=[
        'not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor)',
        'any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda for t in (x, weight))',
        'x.device != weight.device',
        'not isinstance(normalized_shape, (tuple, list)) or any(type(d) is not int or d < 0 for d in normalized_shape)',
        'len(normalized_shape)>x.ndim or tuple(x.shape[x.ndim-len(normalized_shape):]) != tuple(normalized_shape) or tuple(weight.shape) != tuple(normalized_shape) or math.prod(normalized_shape)<=0',
        'any(t.is_neg() or t.is_conj() for t in (x, weight))',
        'type(eps) is not float or not math.isfinite(eps) or eps<=0.0',
        'M>=2**31 or N>=2**31','BLOCK_SIZE>=2**31 or max(0,M-1)*N>=2**31',
        'any(t.data_ptr()%4 != 0 for t in (x, weight, y))',
        'x.data_ptr()+4*max_offset>=2**64 or weight.data_ptr()+4*(BLOCK_SIZE-1)>=2**64 or y.data_ptr()+4*max_offset>=2**64']
    assert guards==list(map(expr,expected))
    attrs=[n.attr for n in ast.walk(fn) if isinstance(n,ast.Attribute)]
    assert attrs.count('contiguous')==2 and not set(attrs)&{'clone','resolve_neg','copy_'}
    contract=(HERE/'Contract.lean').read_text()
    for field in ['input','weight','device','normalizedType','rank','suffix','weightShape','positive','flags',
                  'epsilonType','epsilonFinite','epsilonPositive','rowsBound','columnsBound','blockBound','rowProducts']:
        assert f'  {field} :' in contract
    for n in list(range(1,4097))+[2**p+d for p in range(12,31) for d in [-1,0,1]]:
        x=n-1
        for shift in [1,2,4,8,16,32]:x|=x>>shift
        assert x+1==triton.next_power_of_2(n)

class TensorFacade:
    events=[]
    def __init__(self,data,*,shape=None,cuda=True,device='cuda:0',conj=False,pointer=None):
        self.data=data; self.shape=tuple(data.shape) if shape is None else shape
        self.ndim=len(self.shape);self.dtype=data.dtype;self.layout=data.layout
        self.is_cuda=cuda;self.device=torch.device(device) if cuda else torch.device('cpu')
        self.conj=conj;self.pointer=pointer
    def is_neg(self):return self.data.is_neg()
    def is_conj(self):return self.conj or self.data.is_conj()
    def data_ptr(self):return self.data.data_ptr() if self.pointer is None else self.pointer
    def contiguous(self):
        self.events.append(('copy',self))
        return self if self.data.is_contiguous() else TensorFacade(self.data.contiguous(),shape=self.shape,device=self.device)

def host_checks(kernel):
    events=TensorFacade.events;allocations=[];launches=[];current=[torch.device('cuda:0')];inject={}
    @contextlib.contextmanager
    def device_context(device):
        before=current[0];current[0]=device
        try:yield
        finally:current[0]=before
    def empty_like(x):
        out=TensorFacade(torch.empty_like(x.data),shape=x.shape,device=x.device,**inject)
        events.append(('allocate',out));allocations.append(out);return out
    class Launch:
        def __getitem__(self,grid):
            def invoke(y,x,w,*args):
                events.append(('launch',y));launches.append(y)
                assert current[0]==x.device==w.device==y.device
                N=args[0]; assert args[:5]==(N,1,N,1,N)
                assert x.data.is_contiguous() and w.data.is_contiguous()
                if x.shape!=tuple(x.data.shape):return
                kernel[grid](y.data,x.data,w.data,*args)
            return invoke
    namespace={'torch':SimpleNamespace(Tensor=TensorFacade,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=device_context)),
        'triton':triton,'math':math,'rms_norm_kernel':Launch()}
    fn=node(SOURCE,'forward');fn.decorator_list=[]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),namespace)
    invoke=lambda x,n,w,eps=1e-5:namespace['forward'](None,x,n,w,eps)
    X=TensorFacade;x=torch.ones(2,3,4);w=torch.ones(4)
    bad=[(object(),(4,),X(w)),(X(x),(4,),object()),(X(x.half()),(4,),X(w)),(X(x),(4,),X(w.half())),
         (X(x,cuda=False),(4,),X(w)),(X(x),(4,),X(w,cuda=False)),(X(x),(4,),X(w,device='cuda:1')),
         (X(x),4,X(w)),(X(x),(True,),X(torch.ones(1))),(X(x),(-1,),X(w)),(X(x),(4.,),X(w)),
         (X(x),(1,2,3,4),X(w)),(X(x),(3,),X(torch.ones(3))),(X(x),(4,),X(torch.ones(3))),
         (X(torch.empty(2,0)),(0,),X(torch.empty(0))),
         (X(torch._neg_view(x)),(4,),X(w)),(X(x),(4,),X(torch._neg_view(w))),
         (X(x,conj=True),(4,),X(w)),(X(x),(4,),X(w,conj=True)),
         (X(torch.ones(1,1),shape=(2**31,1)),(1,),X(torch.ones(1))),
         (X(torch.ones(1,1),shape=(1,2**31)),(2**31,),X(torch.ones(1),shape=(2**31,))),
         (X(torch.ones(1,1),shape=(1,2**30+1)),(2**30+1,),X(torch.ones(1),shape=(2**30+1,))),
         (X(torch.ones(1,4),shape=(2**29+1,4)),(4,),X(w))]
    for args in bad:
        before=len(events)
        try:invoke(*args)
        except (TypeError,ValueError):pass
        else:raise AssertionError('expected pre-copy rejection')
        assert len(events)==before
    for eps in [1,0.,-1.,float('nan'),float('inf')]:
        try:invoke(X(x),(4,),X(w),eps)
        except ValueError:pass
        else:raise AssertionError('expected epsilon rejection')
    for arg in [0,1,2]:
        for pointer in [1,2**64-4]:
            xx=X(x,pointer=pointer if arg==0 else None);ww=X(w,pointer=pointer if arg==1 else None)
            if arg==2:inject['pointer']=pointer
            before=len(launches)
            try:invoke(xx,(4,),ww)
            except ValueError:pass
            else:raise AssertionError('expected post-copy pointer rejection')
            assert len(launches)==before
            inject.clear()
    admitted=[(x,(4,),w),(x.transpose(0,1),(4,),w),
              (torch.ones(1,3,4).expand(2,3,4),(4,),torch.ones(1).expand(4)),
              (torch.ones(28)[4:].reshape(2,3,4),(4,),torch.ones(8)[4:]),
              (x,(3,4),torch.ones(3,4).t().contiguous().t()),
              (torch.empty(0,3,4),(4,),w),(torch.ones(2,3),(),torch.tensor(1.)),
              (torch.full((2,3,4),float('nan')),(4,),w)]
    for data,n,weight in admitted:
        xx=X(data,device='cuda:1');ww=X(weight,device='cuda:1');before=len(events)
        result=invoke(xx,n,ww)
        assert result is allocations[-1] and result.shape==xx.shape and result.device==xx.device
        assert [e[0] for e in events[before:]]==['copy','copy','allocate','launch']
        assert current[0]==torch.device('cuda:0')
    invoke(X(torch.ones(1,1),shape=(2**31-1,1)),(1,),X(torch.ones(1)))
    return len(bad)+11,len(admitted)+1

def interpreter_checks(kernel):
    count=0
    for M,N in [(1,1),(2,3),(3,31),(2,32),(1,33),(1,513),(2,4097)]:
        x=torch.ones(M,N);x.reshape(-1)[::2]=-1;w=torch.full((N,),2.);y=torch.empty_like(x)
        kernel[(M,)](y,x,w,N,1,N,1,N,3.,triton.next_power_of_2(N))
        torch.testing.assert_close(y,x,rtol=0,atol=0);count+=1
    # Original flattened row offset can wrap despite legal grid-x.
    assert (2**29)*4==2**31
    return count

if __name__=='__main__':
    source_checks()
    spec=importlib.util.spec_from_file_location('rms_norm_checked',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rejected,admitted=host_checks(module.rms_norm_kernel)
    controls=interpreter_checks(module.rms_norm_kernel)
    print(f'PASS: unchanged kernel AST, exact guards and copy order; {rejected} rejects; {admitted} admits; {controls} exact CPU interpreter controls')
    print(f'PASS: row-product overflow witness; torch={torch.__version__}; Triton={triton.__version__}; CUDA={torch.cuda.is_available()}; no GPU execution')

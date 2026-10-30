"""Exact-source CPU and checked metadata controls. No GPU conformance claim."""
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
SOURCE=HERE/'rmsnorm_implementation_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rmsnorm_implementation/rmsnorm_implementation.py'

def node(path,name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
def expr(s): return ast.dump(ast.parse(s,mode='eval').body)
def source_checks():
    assert ast.dump(node(ORIGINAL,'rmsnorm_triton'))==ast.dump(node(SOURCE,'rmsnorm_triton'))
    fn=node(SOURCE,'rmsnorm_wrapper')
    def conditions(body):
        for n in body:
            if isinstance(n,ast.If):
                yield ast.dump(n.test)
                yield from conditions(n.body)
    guards=list(conditions(fn.body))
    expected=[
        'not isinstance(x, torch.Tensor) or not isinstance(rms_weights, torch.Tensor)',
        'any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda for t in (x, rms_weights))',
        'x.device != rms_weights.device',
        'x.ndim != 3 or x.shape[2] <= 0 or rms_weights.shape != (x.shape[2],)',
        'any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (x, rms_weights))',
        'type(eps) is not float or not math.isfinite(eps) or eps <= 0.0',
        'x.numel() != 0', 'batch >= 2**31 or M > 65535 or padded >= 2**31',
        'any(v >= 2**31 for v in (*x.stride(), *rms_weights.stride()))',
        'input_max >= 2**31 or weight_max >= 2**31',
        'x.data_ptr()%4 != 0 or rms_weights.data_ptr()%4 != 0',
        'x.data_ptr()+4*input_max >= 2**64 or rms_weights.data_ptr()+4*weight_max >= 2**64',
        'x.numel() == 0', 'any(v >= 2**31 for v in out.stride()) or output_max >= 2**31',
        'out.data_ptr()%4 != 0 or out.data_ptr()+4*output_max >= 2**64']
    assert guards==list(map(expr,expected))
    attrs={n.attr for n in ast.walk(fn) if isinstance(n,ast.Attribute)}
    assert not attrs & {'contiguous','clone','resolve_neg','copy_'}
    contract=(HERE/'Contract.lean').read_text()
    for name in ['input','weight','device','rank','positive','weightShape','inputDense','weightDense',
                 'epsilonType','epsilonFinite','epsilonPositive','launch','gridX','gridY','loop','strides','offsets','alignment','pointers']:
        assert f'  {name} :' in contract

class TensorFacade:
    def __init__(self,data,*,shape=None,strides=None,cuda=True,device='cuda:0',conj=False,pointer=None):
        self.data=data; self.shape=tuple(data.shape) if shape is None else shape
        self.ndim=len(self.shape); self.dtype=data.dtype; self.layout=data.layout
        self.is_cuda=cuda; self.device=torch.device(device) if cuda else torch.device('cpu')
        self.conj=conj; self.strides=data.stride() if strides is None else strides; self.pointer=pointer
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return self.data.is_contiguous()
    def storage_offset(self): return self.data.storage_offset()
    def is_neg(self): return self.data.is_neg()
    def is_conj(self): return self.conj or self.data.is_conj()
    def stride(self,axis=None): return self.strides if axis is None else self.strides[axis]
    def data_ptr(self): return self.data.data_ptr() if self.pointer is None else self.pointer

def host_checks(kernel):
    allocations=[]; launches=[]; current=[torch.device('cuda:0')]; inject={}
    @contextlib.contextmanager
    def device_context(device):
        before=current[0]; current[0]=device
        try: yield
        finally: current[0]=before
    def empty_like(x):
        out=TensorFacade(torch.empty_like(x.data),shape=x.shape,strides=x.strides,device=x.device,**inject)
        allocations.append(out); return out
    class Launch:
        def __getitem__(self,grid):
            def invoke(x,w,y,*strides,**kw):
                assert current[0]==x.device==w.device==y.device
                assert grid==x.shape[:2] and kw['BLOCK_N_SIZE']==4096 and kw['num_warps']==16
                assert strides==(*x.stride(),*w.stride(),*y.stride())
                launches.append((x,w,y))
                if x.shape != tuple(x.data.shape): return
                kernel[grid](x.data,w.data,y.data,*strides,**kw)
            return invoke
    namespace={'torch':SimpleNamespace(Tensor=TensorFacade,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=device_context)),
        'math':math,'triton':triton,'rmsnorm_triton':Launch()}
    fn=node(SOURCE,'rmsnorm_wrapper')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),namespace)
    invoke=namespace['rmsnorm_wrapper']; X=TensorFacade
    x=torch.ones(2,3,4); w=torch.ones(4)
    bad=[(object(),X(w)),(X(x),object()),(X(x.to(torch.float16)),X(w)),(X(x),X(w.to(torch.float16))),
         (X(x,cuda=False),X(w)),(X(x),X(w,cuda=False)),(X(x),X(w,device='cuda:1')),
         (X(torch.ones(2,4)),X(w)),(X(torch.empty(2,3,0)),X(torch.empty(0))),
         (X(x),X(torch.ones(3))),(X(x.transpose(0,1)),X(w)),(X(x),X(torch.ones(8)[::2])),
         (X(torch._neg_view(x)),X(w)),(X(x),X(torch._neg_view(w))),
         (X(x,conj=True),X(w)),(X(x),X(w,conj=True)),
         (X(x,shape=(2**31,1,4)),X(w)),(X(x,shape=(1,65536,4)),X(w)),
         (X(x,strides=(2**31,4,1)),X(w)),(X(x),X(w,strides=(2**31,))),
         (X(x,strides=(2**30,2**30,1)),X(w)),(X(x),X(w,strides=(2**20,))),
         (X(x,pointer=1),X(w)),(X(x),X(w,pointer=2)),
         (X(x,pointer=2**64-4),X(w)),(X(x),X(w,pointer=2**64-4)),
         (X(torch.ones(1,1,1),shape=(1,1,2147479553),strides=(2147479553,2147479553,1)),
          X(torch.ones(1),shape=(2147479553,),strides=(1,)))]
    for args in bad:
        before=(len(allocations),len(launches))
        try: invoke(*args)
        except (TypeError,ValueError): pass
        else: raise AssertionError('expected rejection')
        assert before==(len(allocations),len(launches))
    for eps in [1,0.,-1.,float('nan'),float('inf')]:
        try: invoke(X(x),X(w),eps)
        except ValueError: pass
        else: raise AssertionError('expected epsilon rejection')
    for pointer in [1,2**64-4]:
        inject['pointer']=pointer; before=len(launches)
        try: invoke(X(x),X(w))
        except ValueError: pass
        else: raise AssertionError('expected output pointer rejection')
        assert len(launches)==before
    inject.clear()
    admitted=[x,torch.empty(0,3,4),torch.empty(2,0,4),
              torch.ones(28)[4:].reshape(2,3,4),
              torch.as_strided(torch.ones(4),(1,1,4),(100000,200000,1)),
              torch.as_strided(torch.ones(6),(2,3,1),(3,1,0)),torch.full((2,3,4),float('nan'))]
    for data in admitted:
        xx=X(data,device='cuda:1'); ww=X(torch.ones(data.shape[-1]),device='cuda:1')
        before=len(launches); result=invoke(xx,ww)
        assert result is allocations[-1] and result.shape==xx.shape and result.device==xx.device
        assert len(launches)==before+(xx.numel()!=0)
        assert current[0]==torch.device('cuda:0')
    n=2147479552
    invoke(X(torch.ones(1,1,1),shape=(1,1,n),strides=(n,n,1)),X(torch.ones(1),shape=(n,),strides=(1,)))
    # Dead metadata needs no signed32 limit when no launch occurs.
    invoke(X(torch.empty(0,1,1),shape=(0,2**31,2**31),strides=(2**70,2**50,1)),
           X(torch.ones(1),shape=(2**31,),strides=(2**60,)))
    return len(bad)+7,len(admitted)+2

def interpreter_checks(kernel):
    cases=[(1,1,1),(1,2,3),(2,3,31),(1,1,4095),(1,2,4096),(2,1,4097),(1,1,8193)]
    for shape in cases:
        x=torch.ones(shape);x.reshape(-1)[::2]=-1
        w=torch.full((shape[-1],),2.);out=torch.empty_like(x)
        kernel[shape[:2]](x,w,out,*x.stride(),*w.stride(),*out.stride(),N_SIZE=shape[-1],eps=3.,BLOCK_N_SIZE=4096,num_warps=16)
        torch.testing.assert_close(out,x,rtol=0,atol=0)
    # Singleton feature axes are contiguous even with zero strides; empty_like
    # preserves that output stride. Only the single active lane may store.
    x=torch.as_strided(torch.tensor([-1.,1.,-1.,1.,-1.,1.]),(2,3,1),(3,1,0))
    w=torch.as_strided(torch.tensor([2.]),(1,),(0,))
    out=torch.empty_like(x)
    assert x.is_contiguous() and w.is_contiguous() and out.stride(2)==0
    kernel[(2,3)](x,w,out,*x.stride(),*w.stride(),*out.stride(),N_SIZE=1,eps=3.,BLOCK_N_SIZE=4096,num_warps=16)
    torch.testing.assert_close(out,x,rtol=0,atol=0)
    assert ((2147479553+4095)//4096)*4096==2**31
    assert int.from_bytes((2**31).to_bytes(4,'little'), 'little',signed=True)==-(2**31)
    return len(cases)+1

if __name__=='__main__':
    source_checks()
    spec=importlib.util.spec_from_file_location('rms_implementation_checked',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rejected,admitted=host_checks(module.rmsnorm_triton)
    controls=interpreter_checks(module.rmsnorm_triton)
    print(f'PASS: unchanged kernel AST; exact ordered guards; {rejected} rejects; {admitted} admits; {controls} exact CPU interpreter controls')
    print(f'PASS: final-loop-increment overflow witness; torch={torch.__version__}; Triton={triton.__version__}; CUDA={torch.cuda.is_available()}; no GPU execution')

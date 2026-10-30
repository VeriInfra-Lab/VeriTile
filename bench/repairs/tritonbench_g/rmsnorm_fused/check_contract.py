"""Checked host metadata and exact-source CPU controls; no native GPU claim."""
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
SOURCE=HERE/'rmsnorm_fused_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/rmsnorm_fused/rmsnorm_fused.py'

def node(path,name):
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)
def expr(text): return ast.dump(ast.parse(text,mode='eval').body)
def source_checks():
    original=node(ORIGINAL,'rms_norm_fwd_fused'); repaired=node(SOURCE,'rms_norm_fwd_fused')
    assert list(map(ast.dump,original.body[1:]))==list(map(ast.dump,repaired.body[1:]))
    assert ast.unparse(repaired.body[0])=='row = tl.program_id(0).to(tl.int64)'
    forward=node(SOURCE,'forward')
    guards=[ast.dump(n.test) for n in forward.body if isinstance(n,ast.If)]
    expected=[
        'not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor)',
        'any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda for t in (x, weight))',
        'x.device != weight.device',
        'x.ndim < 1 or x.shape[-1] <= 0 or weight.shape != (x.shape[-1],)',
        'any(not t.is_contiguous() or t.storage_offset() != 0 or t.is_neg() or t.is_conj() for t in (x, weight))',
        'type(eps) is not float or not math.isfinite(eps) or eps <= 0.0',
        'N > BLOCK_SIZE', 'x.numel() > (2**63 - 1)//4', 'x.numel()//N > 2**31-1',
        'x.numel() != 0 and (x.data_ptr()%4 != 0 or weight.data_ptr()%4 != 0)', 'M == 0']
    assert guards==list(map(expr,expected))
    assert '.Proofs.' not in (HERE/'Kernels.lean').read_text()
    contract=(HERE/'Contract.lean').read_text()
    for field in ['input','weightRepresentation','device','rank','widthPositive','weightShape','inputCanonical',
                  'weightCanonical','epsilonType','epsilonFinite','epsilonPositive','block','byteExtent','grid','alignment']:
        assert f'  {field} :' in contract
    # Check every admitted feature width against the bounded branch definition.
    for n in range(1,16385):
        table=next(2**p for p in range(15) if n<=2**p)
        assert table==min(16384,triton.next_power_of_2(n))
    attrs=[n.attr for n in ast.walk(forward) if isinstance(n,ast.Attribute)]
    assert not set(attrs)&{'contiguous','clone','resolve_neg','copy_'}

class TensorFacade:
    def __init__(self,data,*,shape=None,cuda=True,device='cuda:0',conj=False):
        self.data=data; self.shape=tuple(data.shape) if shape is None else shape
        self.ndim=len(self.shape); self.dtype=data.dtype; self.layout=data.layout
        self.is_cuda=cuda; self.device=torch.device(device) if cuda else torch.device('cpu'); self.conj=conj
    def numel(self): return math.prod(self.shape)
    def is_contiguous(self): return self.data.is_contiguous()
    def storage_offset(self): return self.data.storage_offset()
    def is_neg(self): return self.data.is_neg()
    def is_conj(self): return self.conj or self.data.is_conj()
    def element_size(self): return self.data.element_size()
    def data_ptr(self): return self.data.data_ptr()
    def reshape(self,*shape):
        count=math.prod(self.shape); n=shape[-1]
        shape=(count//n,n)
        data=self.data.reshape(-1,self.data.shape[-1])
        return TensorFacade(data,shape=shape,device=self.device)

def load_kernel():
    spec=importlib.util.spec_from_file_location('rms_fused_checked',SOURCE)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.rms_norm_fwd_fused

def host_checks(kernel):
    allocations=[]; launches=[]; current=[torch.device('cuda:0')]
    @contextlib.contextmanager
    def on_device(device):
        before=current[0]; current[0]=device
        try: yield
        finally: current[0]=before
    def empty_like(x,memory_format):
        assert memory_format==torch.contiguous_format
        out=TensorFacade(torch.empty_like(x.data,memory_format=memory_format),shape=x.shape,device=x.device)
        allocations.append(out); return out
    class Launch:
        def __getitem__(self,grid):
            def invoke(x,y,w,stride,n,eps,**kwargs):
                assert current[0]==x.device==y.device==w.device
                assert stride==n==x.shape[-1]
                launches.append((x,y,w,grid))
                if tuple(x.shape)!=tuple(x.data.shape): return
                kernel[grid](x.data,y.data,w.data,stride,n,eps,**kwargs)
            return invoke
    fn=node(SOURCE,'forward')
    namespace={'torch':SimpleNamespace(Tensor=TensorFacade,strided=torch.strided,float32=torch.float32,
        contiguous_format=torch.contiguous_format,empty_like=empty_like,cuda=SimpleNamespace(device=on_device)),
        'math':math,'triton':triton,'rms_norm_fwd_fused':Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(SOURCE),'exec'),namespace)
    def invoke(x,w,eps=1e-6): return namespace['forward'](SimpleNamespace(weight=w,variance_epsilon=eps),x)
    x=torch.ones(2,4); w=torch.ones(4)
    X=lambda a,**kw:TensorFacade(a,**kw)
    cases=[(object(),X(w)),(X(x),object()),(X(x.to(torch.float16)),X(w)),(X(x),X(w.to(torch.float16))),
        (X(x,cuda=False),X(w)),(X(x),X(w,cuda=False)),(X(x),X(w,device='cuda:1')),
        (X(torch.tensor(1.)),X(torch.ones(1))),(X(torch.empty(2,0)),X(torch.empty(0))),
        (X(x),X(torch.ones(3))),(X(x.t()),X(torch.ones(2))),
        (X(x),X(torch.ones(8)[::2])),(X(torch.ones(12)[4:].reshape(2,4)),X(w)),
        (X(x),X(torch.ones(8)[4:])),(X(torch._neg_view(x)),X(w)),(X(x),X(torch._neg_view(w))),
        (X(x,conj=True),X(w)),(X(x),X(w,conj=True)),
        (X(torch.ones(1,16385)),X(torch.ones(16385))),
        (X(torch.ones(1,1),shape=(2**62,1)),X(torch.ones(1))),
        (X(torch.ones(1,1),shape=(2**31,1)),X(torch.ones(1)))]
    for offset in (1,2,3):
        storage=torch.arange(40,dtype=torch.uint8).untyped_storage()[offset:offset+32]
        data=torch.empty(0,dtype=torch.float32).set_(storage,0,(2,4),(4,1))
        cases.append((X(data),X(w)))
        dataw=torch.empty(0,dtype=torch.float32).set_(storage,0,(4,),(1,))
        cases.append((X(x),X(dataw)))
    for args in cases:
        before=(len(allocations),len(launches))
        try: invoke(*args)
        except (TypeError,ValueError,RuntimeError): pass
        else: raise AssertionError('expected rejection')
        assert before==(len(allocations),len(launches))
    for eps in [1,float('nan'),float('inf'),0.,-1.]:
        try: invoke(X(x),X(w),eps)
        except ValueError: pass
        else: raise AssertionError('expected epsilon rejection')
    admitted=[torch.ones(4),x,torch.ones(1,1,4),torch.empty(0,4),
              torch.as_strided(torch.ones(4),(1,4),(100000,1)),torch.full((2,4),float('nan'))]
    for data in admitted:
        xx=X(data,device='cuda:1'); ww=X(w,device='cuda:1')
        before=len(launches); result=invoke(xx,ww)
        assert result is allocations[-1] and result.shape==xx.shape and result.device==xx.device
        assert len(launches)==before+(xx.numel()!=0)
        assert current[0]==torch.device('cuda:0')
    invoke(X(torch.ones(1,1),shape=(2**31-1,1)),X(torch.ones(1)))
    return len(cases)+5,len(admitted)+1

def interpreter_checks(kernel):
    count=0
    for shape in [(1,),(2,2),(2,3),(1,31),(2,32),(2,33),(1,127),(2,128),(1,129),(2,3,513),(1,4096),(1,16384)]:
        n=shape[-1]; data=torch.ones(shape); data.reshape(-1)[::2]=-1
        w=torch.full((n,),2.); out=torch.empty_like(data); block=triton.next_power_of_2(n)
        kernel[(data.numel()//n,)](data.reshape(-1,n),out,w,n,n,3.,BLOCK_SIZE=block,
                                  num_warps=min(max(block//256,1),8))
        torch.testing.assert_close(out,data,rtol=0,atol=0); count+=1
    # Original passes input's gapped row stride to an independently packed output.
    x=torch.ones(2,4)[:,::2]; y=torch.empty_like(x)
    assert x.stride(0)==4 and y.stride(0)==2 and 1*x.stride(0)+1>=y.numel()
    return count

if __name__=='__main__':
    source_checks(); kernel=load_kernel(); rejected,admitted=host_checks(kernel); controls=interpreter_checks(kernel)
    print(f'PASS: exact source/guard binding; {rejected} rejects; {admitted} admits; {controls} CPU interpreter controls')
    print(f'PASS: original gapped-output counterexample; torch={torch.__version__}; CUDA={torch.cuda.is_available()}; no GPU execution')

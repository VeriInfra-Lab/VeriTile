"""Direct contiguous SwiGLU source/host replay with CPU physical storage."""
import ast
import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'swiglu_triton_repaired.py'
ORIGINAL=HERE.parents[2]/'tritonbench_g/swiglu_triton/swiglu_triton.py'
spec=importlib.util.spec_from_file_location('swiglu_fixtures',HERE.parent/'swiglu_fwd/check_contract.py')
f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
Pointer,storage,function=f.Pointer,f.storage,f.function

class Tensor(f.Tensor):
    def wrap(self,value): return Tensor(value,self.device,self.is_cuda)
    def view(self,*shape): return self.wrap(self.data.view(*shape))
    def is_contiguous(self): return self.data.is_contiguous()
    def numel(self): return self.data.numel()


def interpreter():
    pid=[0]
    def load(ptr,mask,other):
        value=torch.full(mask.shape,other,dtype=torch.float32)
        value[mask]=storage(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offset[mask]]
        return value
    def store(ptr,value,mask):
        storage(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offset[mask]]=value[mask]
    ns={'tl':SimpleNamespace(program_id=lambda _:torch.tensor(pid[0]),arange=torch.arange,
        int64=torch.int64,float32=torch.float32,load=load,store=store,sigmoid=torch.sigmoid)}
    nodes=[function(SOURCE,name) for name in ('silu','_swiglu_forward_kernel','_swiglu_backward_kernel')]
    for node in nodes:
        node.decorator_list=[]
        for arg in node.args.args: arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns,pid


def host():
    kernels,pid=interpreter()
    events=[]
    @contextlib.contextmanager
    def device(d):
        events.append(('device',d)); yield
    def empty_like(t):
        result=t.wrap(torch.empty_like(t.data)); events.append(('allocation',result)); return result
    class Launch:
        def __init__(self,name): self.name=name
        def __getitem__(self,grid):
            def run(x,y,z,stride,n_cols,BLOCK_SIZE,num_warps):
                events.append(('launch',self.name,grid,stride,n_cols,BLOCK_SIZE,num_warps))
                for row in reversed(range(grid[0])):
                    pid[0]=row
                    kernels[self.name](*(Pointer(t.data) for t in (x,y,z)),stride,n_cols,BLOCK_SIZE)
            return run
    ns={'torch':SimpleNamespace(Tensor=Tensor,strided=torch.strided,float32=torch.float32,
        empty_like=empty_like,cuda=SimpleNamespace(device=device),version=SimpleNamespace(hip=None)),
        'triton':SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length())}
    for name in ('_swiglu_forward_kernel','_swiglu_backward_kernel'): ns[name]=Launch(name)
    nodes=[function(SOURCE,name) for name in ('is_hip','calculate_settings','_check_tensor','_check_call','swiglu_forward','swiglu_backward')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns,events


def check_source():
    for name in ('is_hip','calculate_settings','silu','_swiglu_forward_kernel','_swiglu_backward_kernel'):
        assert ast.dump(function(SOURCE,name))==ast.dump(function(ORIGINAL,name))
    calls=[ast.unparse(n.func) for n in ast.walk(ast.parse(SOURCE.read_text())) if isinstance(n,ast.Call)]
    assert not any(c.endswith(('.contiguous','.clone','.copy_','.resolve_neg')) for c in calls)
    assert 'cols cols block' in (HERE/'Kernels.lean').read_text()
    assert 'inputCheck_eq' in (HERE/'Proofs/InputGuards.lean').read_text()
    selected = (HERE/'Proofs/InputBinding.lean').read_text()
    assert 'InputAdmitted p ∧ blockSize p.cols ≤ 65536' in selected
    assert 'inputCheck p && decide (blockSize p.cols ≤ 65536)' in selected
    tests = [ast.unparse(n.test) for n in function(SOURCE, '_check_call').body if isinstance(n, ast.If)]
    assert tests == ['shaped.ndim == 0 or shaped.shape[-1] == 0',
        'any((t.device != shaped.device or t.numel() != n for t in tensors))',
        'rows >= 2 ** 31 or n + 65536 >= 2 ** 63', 'n > 0']


def check_host():
    ns,events=host(); forward,backward=ns['swiglu_forward'],ns['swiglu_backward']
    torch.manual_seed(53)
    shapes=[(3,1),(2,33),(2,127),(2,128),(2,129),(2,3,257),(1,1025),(1,65536),(0,33)]
    for shape in shapes:
        n=torch.empty(shape).numel()
        ab=torch.randn(n+9); bb=torch.randn(n+11); db=torch.randn(n+13)
        a,b,dc=Tensor(ab[3:3+n].reshape(shape)),Tensor(bb[5:5+n].reshape(shape)),Tensor(db[7:7+n].reshape(shape))
        beforeA,beforeB,beforeD=ab.clone(),bb.clone(),db.clone()
        x,y,d=a.data.clone(),b.data.clone(),dc.data.clone()
        events.clear()
        av,bv,c=forward(a,b)
        torch.testing.assert_close(c.data,x*torch.sigmoid(x)*y,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(ab,beforeA,rtol=0,atol=0)
        torch.testing.assert_close(bb,beforeB,rtol=0,atol=0)
        assert tuple(av.shape)==(n//shape[-1],shape[-1]) and bv.shape==av.shape
        assert av.untyped_storage().data_ptr()==a.untyped_storage().data_ptr()
        assert bv.untyped_storage().data_ptr()==b.untyped_storage().data_ptr()
        assert c is not a and c is not b and tuple(c.shape)==shape
        assert any(e[0]=='device' for e in events)
        assert sum(e[0]=='launch' for e in events)==bool(n)
        events.clear()
        da,dbout=backward(av,bv,dc)
        sig=torch.sigmoid(x); silu=x*sig
        torch.testing.assert_close(da.data,d*(silu*(1-sig)+sig)*y,rtol=2e-6,atol=1e-6)
        torch.testing.assert_close(dbout.data,d*silu,rtol=1e-6,atol=1e-7)
        assert tuple(da.shape)==shape and tuple(dbout.shape)==shape
        assert da.untyped_storage().data_ptr()==a.untyped_storage().data_ptr()
        assert dbout.untyped_storage().data_ptr()==b.untyped_storage().data_ptr()
        torch.testing.assert_close(db,beforeD,rtol=0,atol=0)
        assert torch.equal(ab[:3],beforeA[:3]) and torch.equal(ab[3+n:],beforeA[3+n:])
        assert torch.equal(bb[:5],beforeB[:5]) and torch.equal(bb[5+n:],beforeB[5+n:])
        assert sum(e[0]=='launch' for e in events)==bool(n)
    a,b=Tensor(torch.randn(2,8)),Tensor(torch.randn(2,8))
    failures=[([],b,TypeError),(Tensor(torch.randn(2,8,dtype=torch.float64)),b,TypeError),
        (Tensor(torch.randn(2,16)[:,::2]),b,ValueError),(Tensor(torch._neg_view(a.data)),b,ValueError),
        (Tensor(a.data,'cpu',False),b,ValueError),(a,Tensor(b.data,'cuda:0'),ValueError),
        (a,Tensor(torch.randn(4,4)),ValueError),(a,a,ValueError),
        (Tensor(torch.empty(1<<31,1,device='meta')),Tensor(torch.empty(1<<31,1,device='meta')),ValueError),
        (Tensor(torch.empty(1,65537)),Tensor(torch.empty(1,65537)),RuntimeError)]
    for x,y,error in failures:
        events.clear()
        try: forward(x,y)
        except error: pass
        else: raise AssertionError('unsupported input admitted')
        assert not any(e[0]=='launch' for e in events)
    for hip in (None,'test'):
        ns['torch'].version.hip=hip
        for width,block,warps in ((1,1,4),(2047,2048,8),(8191,8192,16),(32767,32768,16 if hip else 32),(65536,65536,16 if hip else 32)):
            assert ns['calculate_settings'](width)==(block,warps)
    print(f'Unchanged JIT/settings AST; {len(shapes)} forward and in-place backward paths; {len(failures)} rejections; CUDA/HIP warp branches passed.')
    print(f'CPU physical-storage replay only; CUDA mocked. CUDA available: {torch.cuda.is_available()}.')

def check_original_counterexamples():
    kernels, pid = interpreter()
    # Source uses one shared row stride; a strided B view reads padding as data.
    a = torch.ones(2, 3)
    b = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, :3]
    out = torch.empty_like(a)
    for row in range(2):
        pid[0] = row
        kernels['_swiglu_forward_kernel'](Pointer(a), Pointer(b), Pointer(out), 3, 3, 4)
    assert not torch.allclose(out, a * torch.sigmoid(a) * b)
    # In backward an aliased A/B pair cannot retain both distinct gradients.
    x, dc = torch.ones(1, 1), torch.ones(1, 1)
    sig = torch.sigmoid(x.clone()); expected_da = (sig * (1 - sig) + sig)
    pid[0] = 0
    kernels['_swiglu_backward_kernel'](Pointer(dc), Pointer(x), Pointer(x), 1, 1, 1)
    assert not torch.allclose(x, expected_da)
    print('Original counterexamples retained: shared-stride mismatch and aliased backward outputs.')


if __name__=='__main__':
    check_source(); check_host(); check_original_counterexamples()

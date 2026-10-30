"""Check the revised sin_computation source and host over CPU backing storage."""
import ast
import contextlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'sin_computation_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/sin_computation/sin_computation.py'
spec = importlib.util.spec_from_file_location('tensor_test_helpers', HERE.parent / 'add_example/check_contract.py')
helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
TensorFacade, Pointer = helpers.TensorFacade, helpers.Pointer
raw_storage, unaligned_tensor = helpers.raw_storage, helpers.unaligned_tensor


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def check_source():
    assert ast.dump(function(SOURCE, 'sin_kernel')) == ast.dump(function(ORIGINAL, 'sin_kernel'))
    wrapper = function(SOURCE, 'sin_triton_repaired')
    expected = ['not isinstance(x, torch.Tensor) or not isinstance(out, torch.Tensor)',
        'x.layout != torch.strided or out.layout != torch.strided',
        'x.dtype != torch.float32 or out.dtype != torch.float32',
        'not x.is_cuda or not out.is_cuda', 'x.device != out.device',
        'out.numel() < n_elements', 'not x.is_contiguous() or not out.is_contiguous()',
        'x.is_neg() or out.is_neg()', 'n_elements >= 2**31', 'n_elements == 0',
        'xp % 4 != 0 or op % 4 != 0',
        'xp != op and xp + 4*n_elements > op and op + 4*n_elements > xp']
    guards = [n.test for n in wrapper.body if isinstance(n, ast.If)]
    assert list(map(ast.dump, guards)) == [ast.dump(ast.parse(e, mode='eval').body) for e in expected]
    lean = (HERE / 'Contract.lean').read_text()
    expected_lean = ['x.layout != .strided || out.layout != .strided',
        'x.dtype != .floating .fp32 || out.dtype != .floating .fp32',
        '!(x.isCuda && out.isCuda)', 'x.device != out.device', 'out.numel < x.numel',
        '!(x.isContiguous && out.isContiguous)', 'x.isNeg || out.isNeg', 'x.numel >= 2^31',
        'x.numel > 0 && (x.dataPtr.toNat % 4 != 0 || out.dataPtr.toNat % 4 != 0)',
        'x.numel > 0 && !decide (Compatible x out)']
    checks = lean.split('def checkTensorInputs',1)[1].split('def checkInputs',1)[0]
    assert re.findall(r'\bif (.*?) then', checks) == expected_lean
    assert 'x.dataPtr = out.dataPtr ∨' in lean
    assert 'x.dataPtr.toNat + 4*x.numel ≤ out.dataPtr.toNat ∨' in lean
    assert 'out.dataPtr.toNat + 4*x.numel ≤ x.dataPtr.toNat' in lean
    for field in ('layout','dtype','cuda','device','capacity','contiguous','noNeg','extent','alignment','overlap'):
        assert re.search(r'^  '+field+r' :',lean,re.M)
    text = SOURCE.read_text()
    assert '((n_elements + 3) // 4,)' in text and 'BLOCK_SIZE=4' in text
    assert not re.search(r'\.(clone|resolve_neg|contiguous|copy_|view)\(',text)
    assert not re.search(r'^import .*Proofs', (HERE/'Kernels.lean').read_text(),re.M)
    print('Unchanged kernel AST, exact guard mapping, prefix contract and minimal grid passed.')


def interpreter(kernel_name='sin_kernel'):
    pid = [0]
    def load(ptr, mask):
        vals = torch.full_like(ptr.offsets,float('nan'),dtype=torch.float32)
        vals[mask] = raw_storage(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offsets[mask].long()]
        return vals
    def store(ptr, vals, mask):
        raw_storage(ptr.tensor)[ptr.tensor.storage_offset()+ptr.offsets[mask].long()] = vals[mask]
    node = function(SOURCE,kernel_name); node.decorator_list=[]
    for a in node.args.args: a.annotation=None
    ns={'tl':SimpleNamespace(program_id=lambda axis:pid[0],
        arange=lambda a,b:torch.arange(a,b,dtype=torch.int32),load=load,store=store,sin=torch.sin)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec'),ns)
    return ns[kernel_name],pid


def counterexamples(kernel,pid):
    x=torch.tensor([0.,0.,1.,0.])[::2]; out=torch.empty(2)
    pid[0]=0; kernel(Pointer(x),Pointer(out),2,4)
    assert out[1]==0 and not torch.equal(out,torch.sin(x))
    neg=torch._neg_view(torch.ones(2)); kernel(Pointer(neg),Pointer(out),2,4)
    assert not torch.equal(out,torch.sin(neg))
    storage=torch.arange(8,dtype=torch.float32)/10
    original=storage.clone(); x=storage[:6];out=storage[1:7]
    for p in range(2):pid[0]=p;kernel(Pointer(x),Pointer(out),6,4)
    assert not torch.equal(out,torch.sin(original[:6]))
    def s32(n):return (n+2**31)%2**32-2**31
    n=2**29+1; wrapped=s32((2**29)*4)
    assert wrapped==-2**31 and wrapped<n
    assert (n+3)//4 < n
    for n in (2**29+1,2**31-1):
        last=((n+3)//4-1)*4+3
        assert 0<=last<2**31
    print('Original stride2/negative/shifted-overlap/overlaunch failures retained; repaired extent boundary checked.')


def check_host(kernel,pid):
    spec=importlib.util.spec_from_file_location('sin_computation_contract',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    calls=[]
    class Launch:
        def __getitem__(self,grid):
            def run(x,out,n,BLOCK_SIZE):
                assert BLOCK_SIZE==4 and grid==((n+3)//4,)
                calls.append((x,out,n,grid))
                if x.data.device.type!='meta':
                    for p in reversed(range(grid[0])):
                        pid[0]=p;kernel(Pointer(x.data),Pointer(out.data),n,BLOCK_SIZE)
            return run
    module.torch=SimpleNamespace(Tensor=TensorFacade,strided=torch.strided,float32=torch.float32,
        cuda=SimpleNamespace(device=lambda _:contextlib.nullcontext()))
    module.sin_kernel=Launch()
    valid=TensorFacade(torch.ones(3)); output=TensorFacade(torch.empty(3))
    bad=[([],output,TypeError),(valid,[],TypeError),
        (TensorFacade(torch.eye(2).to_sparse()),output,TypeError),
        (TensorFacade(torch.ones(3,dtype=torch.float64)),output,TypeError),
        (valid,TensorFacade(torch.empty(3,dtype=torch.float64)),TypeError),
        (TensorFacade(torch.ones(3),device='cpu',cuda=False),output,ValueError),
        (valid,TensorFacade(torch.empty(3),device='cuda:1'),ValueError),
        (valid,TensorFacade(torch.empty(2)),ValueError),
        (TensorFacade(torch.ones(6)[::2]),output,ValueError),
        (valid,TensorFacade(torch.empty(6)[::2]),ValueError),
        (TensorFacade(torch._neg_view(torch.ones(3))),output,ValueError),
        (valid,TensorFacade(torch._neg_view(torch.empty(3))),ValueError),
        (TensorFacade(torch.empty(2**31,device='meta')),TensorFacade(torch.empty(2**31,device='meta')),ValueError)]
    storage=torch.zeros(8)
    bad += [(TensorFacade(storage[:6]),TensorFacade(storage[1:7]),ValueError),
        (TensorFacade(storage[1:7]),TensorFacade(storage[:6]),ValueError)]
    for k in (1,2,3):
        bad += [(TensorFacade(unaligned_tensor(k)),TensorFacade(torch.empty(4)),ValueError),
            (TensorFacade(torch.ones(4)),TensorFacade(unaligned_tensor(k)),ValueError)]
    for x,out,error in bad:
        calls.clear()
        try:module.sin_triton_repaired(x,out)
        except error:pass
        else:raise AssertionError('unsupported input admitted')
        assert not calls
    pairs=[(torch.arange(n,dtype=torch.float32)/10,torch.full((n+3,),17.)) for n in (1,3,4,5,9)]
    pairs += [(torch.tensor(2.),torch.full((2,2),17.)),(torch.arange(3,dtype=torch.float32),torch.full((2,4),17.)),
        (torch.empty(0),torch.full((5,),17.)),(unaligned_tensor(1,empty=True),torch.empty(0))]
    b=torch.arange(20,dtype=torch.float32)/10
    pairs += [(b[2:7],torch.full((9,),17.)[1:8]),(b[:5],b[:8]),(b[6:9],b[:12]),(b[:3],b[8:15])]
    exceptional=torch.tensor([0.,-0.,float('nan'),float('inf'),-float('inf'),1e-40])
    pairs.append((exceptional,torch.full((8,),17.)))
    for xdata,odata in pairs:
        calls.clear();x=TensorFacade(xdata);out=TensorFacade(odata)
        expected=torch.sin(xdata.clone()).reshape(-1)
        before=list(odata.untyped_storage()); offset=4*odata.storage_offset();size=4*xdata.numel()
        identity=(out.data,out.shape,out.device,out.data.storage_offset(),out.data.data_ptr())
        result=module.sin_triton_repaired(x,out)
        assert result is None
        assert identity==(out.data,out.shape,out.device,out.data.storage_offset(),out.data.data_ptr())
        torch.testing.assert_close(odata.reshape(-1)[:xdata.numel()],expected,rtol=0,atol=0,equal_nan=True)
        after=list(odata.untyped_storage())
        assert after[:offset]==before[:offset] and after[offset+size:]==before[offset+size:]
        assert len(calls)==(1 if xdata.numel() else 0)
        if calls:assert calls[0][0] is x and calls[0][1] is out
    print(f'{len(bad)} rejection and {len(pairs)} admission cases passed; exact aliases, prefix tail and exterior bytes covered.')



def check_candidate_binding():
    original = function(SOURCE, "sin_kernel")
    candidate = function(SOURCE, "sin_kernel_index_commuted")
    expected = ast.parse(ast.unparse(original)).body[0]
    expected.name = "sin_kernel_index_commuted"
    assignment = next(node for node in expected.body if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "block_start")
    product = assignment.value
    assert isinstance(product, ast.BinOp) and isinstance(product.op, ast.Mult)
    product.left, product.right = product.right, product.left
    assert ast.dump(candidate) == ast.dump(expected)
    canonical = (HERE / "Kernels.lean").read_text()
    assert "def kernel " in canonical and "def indexCommutedKernel " in canonical
    assert "Region .fp32" in canonical and "blockSize" in canonical
    assert ".Proofs." not in canonical
    public = (HERE / "FPEquiv.lean").read_text()
    assert "assumptions : Spec.Assumptions FP.GuardedFragment := []" in public
    assert "domain := []" in public and "Host" not in public
    assert "originalKernel" in (HERE / "Correct.lean").read_text() or "kernel X O n blockSize" in (HERE / "Correct.lean").read_text()

if __name__=='__main__':
    check_source(); check_candidate_binding()
    kernel,pid=interpreter(); counterexamples(kernel,pid);check_host(kernel,pid)
    kernel,pid=interpreter('sin_kernel_index_commuted');check_host(kernel,pid)
    print(f'CUDA available: {torch.cuda.is_available()}; CPU sine validates the interpreter only.')

"""Source/guard checks and CPU Triton-interpreter controls; no GPU claim."""
import ast
import contextlib
import importlib.util
import math
import os
from pathlib import Path
import re
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "chunk_cumsum_kernel_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/chunk_cumsum_kernel/chunk_cumsum_kernel.py"
NAME = "chunk_global_cumsum_scalar_repaired"
KERNEL = "chunk_global_cumsum_scalar_kernel_repaired"
CHECKS = [
    ("s.layout != torch.strided", "s.layout != .strided", "layout"),
    ("s.dtype != torch.float32", "s.dtype != .floating .fp32", "inputDType"),
    ("dtype is not None and dtype != torch.float32", "dtype != none && dtype != some (.floating .fp32)", "outputDType"),
    ("s.ndim != 3", "s.shape.length != 3", "rank"),
    ("not s.is_cuda", "!s.isCuda", "cuda"),
    ("not s.is_contiguous()", "!s.isContiguous", "contiguous"),
    ("s.is_neg()", "s.isNeg", "noNeg"),
    ("s.numel() != 0 and B * H >= 2**31", "s.numel > 0 && rows s >= 2^31", "grid"),
    ("s.numel() != 0 and s.data_ptr() % 4 != 0", "s.numel > 0 && s.dataPtr.toNat % 4 != 0", "alignment"),
]


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def expression(text):
    return ast.dump(ast.parse(text, mode="eval").body)


def source_checks():
    wrapper = function(SOURCE, NAME)
    guards = [ast.dump(n.test) for n in wrapper.body if isinstance(n, ast.If)]
    assert guards == [expression("not isinstance(s, torch.Tensor)"),
                      *(expression(p) for p, _, _ in CHECKS), expression("z.numel() == 0")]
    contract = (HERE / "Contract.lean").read_text()
    body = contract.split("def checkTensorInput", 1)[1].split("def checkInput", 1)[0]
    assert re.findall(r"\bif (.*?) then", body) == [lean for _, lean, _ in CHECKS]
    for _, _, field in CHECKS:
        assert f"  {field} :" in contract
    assert "| .nonTensor => .error .tensor" in contract
    for node in ast.walk(wrapper):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"contiguous", "clone", "resolve_neg"}
    context = next(n for n in wrapper.body if isinstance(n, ast.With))
    assert ast.unparse(context.items[0].context_expr) == "torch.cuda.device(s.device)"
    original = function(ORIGINAL, "chunk_global_cumsum_scalar_kernel")
    repaired = function(SOURCE, KERNEL)
    assert ast.dump(original.decorator_list[0]) == ast.dump(repaired.decorator_list[0])
    old_loop = next(n for n in original.body if isinstance(n, ast.For))
    new_loop = next(n for n in repaired.body if isinstance(n, ast.For))
    # Reduction, carry, scan/sub/add and output conversion remain in source order.
    assert list(map(ast.dump, old_loop.body[3:6])) == list(map(ast.dump, new_loop.body[4:7]))
    old_store, new_store = old_loop.body[-1].value, new_loop.body[-1].value
    assert ast.dump(old_store.args[1]) == ast.dump(new_store.args[1])
    assert ast.unparse(new_loop.body[3]) == "b_s = tl.load(p_s, mask=times < T).to(tl.float32)"
    assert ast.unparse(new_loop.iter) == "range(tl.cast(0, tl.int64), tl.cast(tl.cdiv(T, BT), tl.int64))"
    lean = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in lean
    for line in ["times = i_t * $(BT) + tl.arange(0, BT).to(tl.int64)",
                 "p_s = s+i_bh.to(tl.int64)*$(T)+times",
                 "b_s = tl.load(p_s, mask=times<$(T)).to(tl.float32)",
                 "b_o = tl.cumsum(b_s, axis=0) + b_z[None]"]:
        assert line in lean



class TensorFacade:
    """Real CPU metadata; CUDA ownership is mocked only for wrapper controls."""
    def __init__(self, data, *, cuda=True, shape=None, device="cuda:0"):
        self.data, self.is_cuda = data, cuda
        self.shape = data.shape if shape is None else shape
        self.ndim = len(self.shape)
        self.dtype, self.layout = data.dtype, data.layout
        self.device = torch.device(device) if cuda else torch.device("cpu")
    def numel(self): return math.prod(self.shape)
    def stride(self, axis): return self.data.stride(axis)
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self):
        assert self.numel() > 0, "empty admission must not inspect a pointer"
        return self.data.data_ptr()


def load_interpreter():
    spec = importlib.util.spec_from_file_location("scalar_repair", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.chunk_global_cumsum_scalar_kernel_repaired.fn


def host_checks(kernel):
    allocations, launches = [], []
    current_device = [torch.device("cuda:0")]
    @contextlib.contextmanager
    def on_device(device):
        previous = current_device[0]
        current_device[0] = device
        try: yield
        finally: current_device[0] = previous
    def empty_like(s, dtype):
        out = TensorFacade(torch.empty_like(s.data, dtype=dtype), shape=s.shape, device=s.device)
        allocations.append(out)
        return out
    class Launch:
        def __getitem__(self, grid):
            def invoke(s, z, **kwargs):
                assert current_device[0] == s.device == z.device
                launches.append((s, z, grid, kwargs))
                # Synthetic large shapes test admission only, without allocation.
                if tuple(s.shape) != tuple(s.data.shape): return
                kernel[grid](s.data, z.data, **kwargs, BT=16, num_warps=2)
            return invoke
    node = function(SOURCE, NAME)
    node.returns = None
    for arg in node.args.args: arg.annotation = None
    namespace = {"torch": SimpleNamespace(Tensor=TensorFacade, strided=torch.strided,
        float32=torch.float32, empty_like=empty_like, cuda=SimpleNamespace(device=on_device)), KERNEL: Launch()}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), namespace)
    wrapper = namespace[NAME]
    base = torch.arange(48, dtype=torch.float32)
    unaligned = []
    for offset in (1, 2, 3):
        storage = torch.arange(32, dtype=torch.uint8).untyped_storage()[offset:offset+16]
        unaligned.append(torch.empty(0, dtype=torch.float32).set_(storage, 0, (1, 1, 4), (4, 4, 1)))
        assert unaligned[-1].data_ptr() % 4 == offset
    rejection = [object(), TensorFacade(torch.sparse_coo_tensor(torch.tensor([[0],[0],[0]]), [1.], (1,1,1), check_invariants=True)),
                 TensorFacade(base.to(torch.float16).reshape(2,3,8)), TensorFacade(base),
                 TensorFacade(base.reshape(2,3,8), cuda=False),
                 TensorFacade(base.reshape(2,3,8).transpose(1,2)),
                 TensorFacade(torch._neg_view(base.reshape(2,3,8))),
                 TensorFacade(torch.zeros(1,1,1), shape=(2**31,1,1)),
                 *(TensorFacade(x) for x in unaligned)]
    for s in rejection:
        before = (len(allocations), len(launches))
        try: wrapper(s)
        except (TypeError, ValueError): pass
        else: raise AssertionError("expected rejection")
        assert before == (len(allocations), len(launches))
    try: wrapper(TensorFacade(base.reshape(2,3,8)), torch.float16)
    except TypeError: pass
    else: raise AssertionError("expected output dtype rejection")
    admitted = [base.reshape(2,3,8), base[4:28].reshape(2,3,4),
                torch.as_strided(base, (1,3,4), (99,4,1), 2),
                torch.empty(0,3,4), torch.empty(2,0,4), torch.empty(2,3,0),
                torch.tensor([float("nan"), float("inf")]).reshape(1,1,2)]
    for data in admitted:
        s = TensorFacade(data)
        before = len(launches)
        z = wrapper(s, torch.float32)
        assert z is allocations[-1] and tuple(z.shape) == tuple(s.shape)
        assert z.dtype == torch.float32 and z.device == s.device
        assert len(launches) == before + (s.numel() > 0)
        if s.numel(): assert launches[-1][0] is s and launches[-1][2] == (s.shape[0]*s.shape[1],)
    # Empty tensors need neither usable pointer nor a bounded unused row grid.
    wrapper(TensorFacade(torch.empty(0,1,1), shape=(2**31,1,0)))
    wrapper(TensorFacade(torch.zeros(1,1,1), shape=(2**31-1,1,1)))
    wrapper(TensorFacade(torch.zeros(1,1,1), device="cuda:1"))
    assert current_device[0] == torch.device("cuda:0")
    return len(rejection)+1, len(admitted)+3


def interpreter_checks(kernel):
    count = 0
    for bt, warps in [(16,2),(32,4),(32,2),(64,8),(64,4)]:
        for t in [1,4,15,16,17,31,32,33,63,64,65,129]:
            # Small integers keep the exact Real target representable in FP32.
            storage = (torch.arange(6*t+11, dtype=torch.float32) % 11) - 5
            s = storage[3:3+6*t].reshape(2,3,t)
            z = torch.empty_like(s)
            kernel[(6,)](s,z,T=t,BT=bt,num_warps=warps)
            expected = s.cumsum(-1)
            torch.testing.assert_close(z,expected,rtol=0,atol=0)
            count += 1
    # Original contiguous row addressing disagrees with a transposed logical view.
    view = torch.tensor([1.,2.,3.,4.]).reshape(1,2,2).transpose(1,2)
    assert not view.is_contiguous()
    assert view[0,0].sum().item() == 4 and torch.tensor([1.,2.]).sum().item() == 3
    # Poisoning unspecified tail loads cannot change any active forward prefix.
    from triton.runtime.interpreter import InterpreterBuilder
    old_load = InterpreterBuilder.create_masked_load
    def poison_load(self, ptr, mask, other, *args, **kwargs):
        if other is None:
            import numpy as np
            from triton.runtime.interpreter import TensorHandle
            other = TensorHandle(np.full(ptr.data.shape, np.nan, dtype=np.float32), ptr.dtype.element_ty)
        return old_load(self, ptr, mask, other, *args, **kwargs)
    InterpreterBuilder.create_masked_load = poison_load
    try:
        for bt, warps in [(16,2),(32,4),(32,2),(64,8),(64,4)]:
            s = torch.arange(1,66,dtype=torch.float32).reshape(1,1,65)
            z = torch.empty_like(s)
            kernel[(1,)](s,z,T=65,BT=bt,num_warps=warps)
            torch.testing.assert_close(z,s.cumsum(-1),rtol=0,atol=0)
    finally:
        InterpreterBuilder.create_masked_load = old_load
    # Original row arithmetic wraps even below CUDA's legal grid-x limit.
    assert 33038210 * 65 == 2147483650
    assert ((33038210 * 65 + 2**31) % 2**32) - 2**31 == -2147483646
    return count


if __name__ == "__main__":
    source_checks()
    kernel = load_interpreter()
    rejected, admitted = host_checks(kernel)
    count = interpreter_checks(kernel)
    print(f"PASS: source binding, {rejected} rejections, {admitted} admissions, {count} CPU Triton-interpreter controls")
    print("PASS: five NaN-poison tail controls, original layout and int32-overflow counterexamples")
    print(f"torch={torch.__version__}; CUDA available={torch.cuda.is_available()}; no GPU execution")

"""Targeted source checks on actual CPU storage; CUDA admission and launch are mocked.
The four JIT ASTs execute directly. This is not GPU numerical validation.
Pointer/AST adapters are reused from the previous task's source execution test.
"""
import ast
import contextlib
import copy
import ctypes
from pathlib import Path
import numpy as np
import torch
import triton

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "diag_ssm_triton_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/diag_ssm_triton/diag_ssm_triton.py"

class Pointer:
    def __init__(self, tensor, offset=0):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
        self.tensor = tensor
        self.offset = offset
        available = tensor.untyped_storage().nbytes() - (
            tensor.data_ptr() - tensor.untyped_storage().data_ptr())
        self.raw = np.ctypeslib.as_array(
            (ctypes.c_float * (available // 4)).from_address(tensor.data_ptr()))

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)

    def __sub__(self, offset):
        return self + (-offset)


class TL:
    constexpr = int
    int64 = np.int64

    @staticmethod
    def cast(value, dtype):
        return dtype(value)

    def __init__(self):
        self.pid = 0
        self.accesses = []

    def program_id(self, axis):
        assert axis == 0
        return self.pid

    @staticmethod
    def arange(start, end):
        return np.arange(start, end, dtype=np.int64)

    @staticmethod
    def zeros_like(value):
        return np.zeros_like(value)

    def load(self, ptr, mask, other=0):
        indexes = np.asarray(ptr.offset)[mask]
        assert np.all((0 <= indexes) & (indexes < len(ptr.raw)))
        value = np.full(np.shape(mask), other, dtype=np.float32)
        value[mask] = ptr.raw[indexes]
        self.accesses.append(("load", len(indexes)))
        return value

    def store(self, ptr, value, mask):
        indexes = np.asarray(ptr.offset)[mask]
        assert np.all((0 <= indexes) & (indexes < len(ptr.raw)))
        ptr.raw[indexes] = value[mask]
        self.accesses.append(("store", len(indexes)))


class Launch:
    def __init__(self, kernel, tl):
        self.kernel, self.tl = kernel, tl
        self.calls = []

    def __getitem__(self, grid):
        def run(*args):
            block = args[-1]
            dimensions = grid({"BLOCK_SIZE": block})
            assert block == 128 and len(dimensions) == 1
            tensors = [x for x in args if isinstance(x, torch.Tensor)]
            self.calls.append({
                "kernel": self.kernel.__name__, "grid": dimensions,
                "scalars": list(args[len(tensors):]),
                "tensors": [{"pointer": x.data_ptr(), "shape": list(x.shape),
                    "stride": list(x.stride()), "dtype": str(x.dtype),
                    "is_neg": x.is_neg(), "is_conj": x.is_conj(),
                    "storage_offset": x.storage_offset()} for x in tensors]})
            converted = [Pointer(x) if isinstance(x, torch.Tensor) else x for x in args]
            for pid in range(dimensions[0]):
                self.tl.pid = pid
                self.kernel(*converted)
        return run


class TorchCPU:
    cuda = type("CUDA", (), {"device": staticmethod(lambda _: contextlib.nullcontext())})()
    def __getattr__(self, name):
        return getattr(torch, name)


def load_source(path):
    tree = ast.parse(path.read_text())
    tl = TL()
    scope = {"torch": TorchCPU(), "triton": triton, "tl": tl}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("diag_ssm_"):
            node.decorator_list = []
            exec(compile(ast.Module([node], []), str(path), "exec"), scope)
            scope[node.name] = Launch(scope[node.name], tl)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("_check"):
            if node.name == "_check_tensor":
                cuda_checks = [n for n in node.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "not tensor.is_cuda"]
                assert len(cuda_checks) == 1
                node.body.remove(cuda_checks[0])
            exec(compile(ast.Module([node], []), str(path), "exec"), scope)
        elif isinstance(node, ast.ClassDef) and node.name == "_ssm_forward":
            exec(compile(ast.Module([node], []), str(path), "exec"), scope)
    return scope, tl


class Context:
    def save_for_backward(self, *values):
        self.saved_tensors = values


def reference(s, x, transition):
    state = s.resolve_conj().resolve_neg().clone()
    rows = []
    for row in x:
        state = state * transition + row
        rows.append(state)
    return torch.stack(rows) if rows else torch.empty_like(x)


def adjoint(s, y, transition, gy):
    carry = torch.zeros_like(s)
    gl = torch.zeros_like(s)
    gx = torch.empty_like(gy)
    for t in reversed(range(len(gy))):
        gx[t] = gy[t] + carry
        previous = s if t == 0 else y[t - 1]
        gl = gl + gx[t] * previous.conj()
        carry = gx[t] * transition.conj()
    return carry, gx, gl.sum(dim=0)


def body_preserved():
    def kernels(path):
        return {n.name: n for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("diag_ssm_")}
    old, new = kernels(ORIGINAL), kernels(CANDIDATE)
    assert old.keys() == new.keys() and len(old) == 4
    for name, kernel in new.items():
        widened = 0
        body = []
        for node in kernel.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "tl.cast":
                target = ast.unparse(node.targets[0])
                assert target in ("length", "batch_size", "dim")
                assert ast.unparse(node.value) == f"tl.cast({target}, tl.int64)"
                widened += 1
                continue
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "col_idx":
                assert ast.unparse(node.value) == "tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE"
                node.value.left = node.value.left.args[0]
                widened += 1
            body.append(node)
        assert widened == 4
        kernel.body = body
        assert ast.dump(kernel, include_attributes=False) == ast.dump(old[name], include_attributes=False)


def rejected(fn, *args):
    try:
        fn(*args)
    except (ValueError, TypeError):
        return
    raise AssertionError("unsupported input was accepted")


def main():
    body_preserved()
    scope, tl = load_source(CANDIDATE)
    host = scope["_ssm_forward"]
    controls = 0
    for dtype in (torch.float32, torch.complex64):
        for length, batch, dim in ((0, 1, 1), (1, 1, 1), (3, 2, 65), (2, 0, 3), (2, 1, 0)):
            s = torch.ones((batch, dim), dtype=dtype)
            x = torch.ones((length, batch, dim), dtype=dtype)
            transition = torch.ones(dim, dtype=dtype)
            gy = torch.ones_like(x)
            context = Context()
            output = host.forward(context, s, x, transition)
            assert context.saved_tensors == (s, output, transition)
            expected = reference(s, x, transition)
            assert torch.equal(output, expected)
            gradients = host.backward(context, gy)
            assert all(torch.equal(a, b) for a, b in zip(gradients, adjoint(s, expected, transition, gy)))
            assert all(t.shape == shape for t, shape in zip(gradients, (s.shape, x.shape, transition.shape)))
            controls += 1
    s = torch.tensor([[1+2j]], dtype=torch.complex64)
    x = torch.tensor([[[2-1j]], [[-1+2j]]], dtype=torch.complex64)
    transition = torch.tensor([1+1j], dtype=torch.complex64)
    gy = torch.tensor([[[1+2j]], [[2-1j]]], dtype=torch.complex64)
    context = Context()
    output = host.forward(context, s, x, transition)
    gradients = host.backward(context, gy)
    inputs = [t.clone().requires_grad_() for t in (s, x, transition)]
    target = torch.autograd.grad(reference(*inputs), inputs, gy)
    assert all(torch.equal(a, b) for a, b in zip(gradients, target))
    # Noncontiguous gradients are the pre-existing backward materialization.
    s = torch.ones((2,3))
    x = torch.ones((2,2,3))
    transition = torch.ones(3)
    context = Context()
    output = host.forward(context, s, x, transition)
    gy = torch.ones((2,3,2)).transpose(1,2)
    assert not gy.is_contiguous()
    assert all(torch.equal(a,b) for a,b in zip(host.backward(context, gy), adjoint(s, output, transition, gy)))
    # Offset/aliased inputs are read-only and preserve padding.
    backing = torch.tensor([99., 1., 99.])
    s = backing[1:2].reshape(1,1)
    transition = backing[1:2]
    context = Context()
    assert host.forward(context, s, torch.zeros((1,1,1)), transition).item() == 1
    assert backing.tolist() == [99.,1.,99.]
    check = scope["_check_inputs"]
    s,x,transition = torch.ones((1,1)),torch.ones((1,1,1)),torch.ones(1)
    rejected(check, None, x, transition)
    rejected(check, s.double(), x.double(), transition.double())
    rejected(check, s, x, torch.ones(1,dtype=torch.complex64))
    rejected(check, torch.ones((2,1)), x, transition)
    rejected(check, s, x.reshape(1,1), transition)
    rejected(check, torch.ones((2,3)).T, torch.ones((1,3,2)), torch.ones(2))
    rejected(check, torch._neg_view(s), x, transition)
    rejected(check, s.to(torch.complex64).conj(), x.to(torch.complex64), transition.to(torch.complex64))
    rejected(host.backward, context, torch._neg_view(x))
    for remainder in (1,2,3):
        raw = torch.empty(32,dtype=torch.uint8)
        bad = torch.empty(0,dtype=torch.float32).set_(raw.untyped_storage()[remainder:],0,(1,1),(1,1))
        assert bad.is_contiguous() and bad.data_ptr()%4 == remainder
        rejected(check, bad, x, transition)
    # The original accepts a contiguous negative view and reads its positive storage.
    original,_ = load_source(ORIGINAL)
    context = Context()
    bad = torch._neg_view(s)
    got = original["_ssm_forward"].forward(context,bad,torch.zeros_like(x),transition)
    assert got.item()==1 and reference(bad,torch.zeros_like(x),transition).item()==-1
    # Source int32 products wrap inside an otherwise representable metadata shape.
    assert ctypes.c_int32((1<<30)*2).value == -(1<<31)
    assert np.int64(1<<30)*np.int64(2) == 1<<31
    print(f"PASS: four JIT bodies preserved; {controls} real/complex shape controls; complex autograd; masks/bounds; offsets/aliasing; guards; original negative-view and int32 overflow counterexamples.")
    print(f"Torch {torch.__version__}; Triton {triton.__version__}; CUDA available: {torch.cuda.is_available()}; GPU execution: no.")


if __name__ == "__main__":
    main()

"""Focused contract/source/host checks. CPU scalar arithmetic is not GPU numerics."""
import ast
import contextlib
import hashlib
import importlib.util
from pathlib import Path
import re
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "l2_norm_triton1_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/l2_norm_triton1/l2_norm_triton1.py"
CHECKS = [
    ("x.layout != torch.strided", "x.layout != .strided", "layout : x.layout=.strided"),
    ("x.dtype != torch.float32", "x.dtype != .floating .fp32", "dtype : x.dtype=.floating .fp32"),
    ("not x.is_cuda", "!x.isCuda", 'cuda : x.device.kind="cuda"'),
    ("x.ndim == 0", "x.shape=[]", "rank : x.shape≠[]"),
    ("N == 0 or N > 16384", "x.shape.getLastD 0=0 || x.shape.getLastD 0>16384",
     "width : 0<x.shape.getLastD 0 ∧ x.shape.getLastD 0≤16384"),
    ("not x.is_contiguous()", "!x.isContiguous", "contiguous : x.isContiguous=Bool.true"),
    ("x.is_neg()", "x.isNeg", "noNeg : x.isNeg=Bool.false"),
    ("x.numel() > 0 and x.data_ptr() % 16 != 0", "x.numel>0 && x.dataPtr.toNat%16 != 0",
     "alignment : 0<x.numel → x.dataPtr.toNat%16=0"),
]


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_binding():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == "3b190a21ae5ef1c37f8d24e16c1757950215749d28a463f3c1989657fa07a017"
    compiler_contract = (HERE / "Proofs/Compilation.lean").read_text()
    pin = re.search(r'def candidateHash : String := "([0-9a-f]+)"', compiler_contract).group(1)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == pin
    kernel = "_l2_norm_fwd_1pass_kernel_repaired"
    assert hashlib.sha256(ast.dump(function(SOURCE, kernel)).encode()).hexdigest() == "804494376c6f21285922e49baa9a22b01131e24e13d4db006b8a52ebeed16571"
    wrapper = function(SOURCE, "_l2_norm_fwd_repaired")
    guards = [node.test for node in wrapper.body if isinstance(node, ast.If)]
    expected = [ast.parse(test, mode="eval").body for test in
                ["not isinstance(x, torch.Tensor)", *(row[0] for row in CHECKS)]]
    assert list(map(ast.dump, guards)) == list(map(ast.dump, expected))
    lean = (HERE / "Contract.lean").read_text()
    contract = lean.split("structure InputContract", 1)[1].split("inductive InputError", 1)[0]
    checks = lean.split("def checkTensorInputs", 1)[1].split("def checkInputs", 1)[0]
    assert re.findall(r"\bif (.*?) then", checks) == [row[1] for row in CHECKS]
    for _, _, premise in CHECKS: assert premise in contract
    assert sum(isinstance(node, ast.Raise) for node in ast.walk(wrapper)) == 9
    calls = [node.func.attr for node in ast.walk(wrapper)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert all(name not in calls for name in ("clone", "contiguous", "resolve_neg", "reshape", "detach"))
    assert calls.count("view") == 2
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()
    print("Original preserved; accepted kernel AST unchanged; 9 guards match Lean; both reshapes are aliasing views.")


class TensorFacade:
    def __init__(self, data, cuda=True):
        self.data, self.is_cuda = data, cuda
        self.device = torch.device("cuda:0" if cuda else "cpu")
    @property
    def layout(self): return self.data.layout
    @property
    def dtype(self): return self.data.dtype
    @property
    def ndim(self): return self.data.ndim
    @property
    def shape(self): return self.data.shape
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()
    def data_ptr(self):
        assert self.numel() > 0
        return self.data.data_ptr()
    def view(self, *shape): return TensorFacade(self.data.view(*shape), self.is_cuda)


def raw_storage(t):
    return torch.empty(0, dtype=torch.float32).set_(
        t.untyped_storage(), 0, (t.untyped_storage().nbytes() // 4,), (1,))


class Pointer:
    def __init__(self, tensor, offsets=0): self.tensor, self.offsets = tensor, offsets
    def __add__(self, offsets): return Pointer(self.tensor, self.offsets + offsets)


def interpreter(path, name):
    pid = [0]
    def load(p, mask, other):
        out = torch.full_like(p.offsets, other, dtype=torch.float32)
        indices = p.tensor.storage_offset() + p.offsets[mask]
        assert torch.all(indices < raw_storage(p.tensor).numel())
        out[mask] = raw_storage(p.tensor)[indices]
        return out
    def store(p, values, mask):
        indices = p.tensor.storage_offset() + p.offsets[mask]
        if not torch.all(indices < raw_storage(p.tensor).numel()):
            raise IndexError("source store leaves fresh output allocation")
        raw_storage(p.tensor)[indices] = values[mask]
    node = function(path, name)
    node.decorator_list = []
    for arg in node.args.args: arg.annotation = None
    namespace = {"tl": SimpleNamespace(program_id=lambda _: torch.tensor(pid[0]),
        arange=torch.arange, load=load, store=store, where=torch.where, float32=torch.float32,
        int64=torch.int64, sqrt=torch.sqrt, sum=lambda x, axis: x.sum(dim=axis))}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name], pid


def host_checks():
    spec = importlib.util.spec_from_file_location("l2_norm_contract_check", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    kernel, pid = interpreter(SOURCE, "_l2_norm_fwd_1pass_kernel_repaired")
    def empty(shape, dtype, device):
        assert dtype == torch.float32
        out = TensorFacade(torch.empty(shape, dtype=dtype))
        calls.append(("allocation", out))
        return out
    class Launch:
        def __getitem__(self, grid):
            def run(x, y, n, eps, block, **options):
                calls.append((grid, x, y, n, eps, block, options))
                for row in range(grid[0]):
                    pid[0] = row
                    kernel(Pointer(x.data), Pointer(y.data), n, eps, block)
            return run
    module.torch = SimpleNamespace(Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
        empty=empty, cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    module._l2_norm_fwd_1pass_kernel_repaired = Launch()
    backing = torch.arange(128, dtype=torch.float32)
    bad = [(None, TypeError), (TensorFacade(torch.eye(2).to_sparse()), TypeError),
           (TensorFacade(torch.ones(3, dtype=torch.float64)), TypeError),
           (TensorFacade(torch.ones(3), cuda=False), ValueError),
           (TensorFacade(torch.tensor(1.)), ValueError),
           (TensorFacade(torch.empty(2, 0)), ValueError),
           (TensorFacade(torch.empty(1, 16385)), ValueError),
           (TensorFacade(torch.ones(4, 8).T), ValueError),
           (TensorFacade(torch.ones(4, 8)[::2]), ValueError),
           (TensorFacade(torch._neg_view(torch.ones(2, 8))), ValueError)]
    for offset in (1, 2, 3):
        data = backing[offset:offset+8]
        assert data.is_contiguous() and data.data_ptr() % 16 == 4*offset
        bad.append((TensorFacade(data), ValueError))
    for x, error in bad:
        calls.clear()
        try: module._l2_norm_fwd_repaired(x)
        except error: pass
        else: raise AssertionError("unsupported input admitted")
        assert not calls
    bytes_storage = torch.arange(80, dtype=torch.uint8).untyped_storage()[1:65]
    empty_unaligned = torch.empty(0, dtype=torch.float32).set_(bytes_storage, 0, (0, 8), (8, 1))
    assert empty_unaligned.untyped_storage().data_ptr() % 16 == 1
    good = [torch.randn(2, n) for n in (1, 3, 16, 32, 128, 129, 16384)] + [
        torch.randn(2, 3, 5), torch.randn(7), torch.empty(0, 8), empty_unaligned, backing[4:36].view(4, 8),
        torch.as_strided(backing, (1, 8), (99, 1), 4),
        torch.tensor([[float("nan"), -0., float("inf"), -float("inf")]])]
    for data in good:
        calls.clear()
        before = raw_storage(data).view(torch.int32).clone()
        x = TensorFacade(data)
        out = module._l2_norm_fwd_repaired(x)
        expected = data * torch.rsqrt((data * data).sum(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(out.data, expected, rtol=3e-6, atol=1e-7, equal_nan=True)
        assert out.shape == x.shape and out.data.is_contiguous()
        assert torch.equal(raw_storage(data).view(torch.int32), before)
        allocation = calls[0][1]
        assert out.data.untyped_storage()._cdata == allocation.data.untyped_storage()._cdata
        if data.numel():
            assert len(calls) == 2
            grid, kernel_x, kernel_y, n, eps, block, options = calls[1]
            assert kernel_x.data.data_ptr() == data.data_ptr()
            assert kernel_x.data.untyped_storage()._cdata == data.untyped_storage()._cdata
            assert kernel_y is allocation and grid == (data.numel() // data.shape[-1],)
            assert n == data.shape[-1] and block == 1 << (n-1).bit_length()
            assert options == {"num_warps": 4, "num_ctas": 1}
        else: assert len(calls) == 1
    old, old_pid = interpreter(ORIGINAL, "_l2_norm_fwd_1pass_kernel")
    x = torch.ones(4, 8)[::2]
    y = torch.full_like(x, -12345.)
    assert x.stride(0) == 16 and y.stride(0) == 8
    old_pid[0] = 1
    try: old(Pointer(x), Pointer(y), x.stride(0), 8, 1e-6, 8)
    except IndexError: pass
    else: raise AssertionError("original stride defect was not reproduced")
    print(f"{len(bad)} rejection cases and {len(good)} direct-view cases passed, including empty rows and offset views.")
    print("Original row-stride out-of-bounds store reproduced. CPU AST/host checks only; no native numerical conformance claim.")


if __name__ == "__main__":
    source_binding()
    host_checks()

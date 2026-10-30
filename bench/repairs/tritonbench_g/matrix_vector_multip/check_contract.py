"""Focused source/host checks with real CPU storage and Triton's interpreter.
CUDA allocation/context and autotune selection are adapted explicitly; no GPU
numerical conformance is claimed.
"""
import ast
import copy
import ctypes
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/veritile-matrix-vector-interpreter")
sys.dont_write_bytecode = True
import torch
import triton

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "matrix_vector_multip_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/matrix_vector_multip/matrix_vector_multip.py"

def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)

def dump(node): return ast.dump(node, include_attributes=False)
def statement(text): return ast.parse(text).body[0]

CHECKS = [
    ("not isinstance(inp, torch.Tensor) or not isinstance(vec, torch.Tensor)", None),
    ("inp.layout != torch.strided or vec.layout != torch.strided", "x.layout != .strided || y.layout != .strided"),
    ("inp.dtype != torch.float32 or vec.dtype != torch.float32", "x.dtype != .floating .fp32 || y.dtype != .floating .fp32"),
    ("not inp.is_cuda or not vec.is_cuda", "!(x.isCuda && y.isCuda)"),
    ("inp.device != vec.device", "x.device != y.device"),
    ("inp.ndim != 2 or vec.ndim != 1", "x.shape.length != 2 || y.shape.length != 1"),
    ("inp.shape[1] != vec.shape[0]", "columns x != vectorLength y"),
    ("not inp.is_contiguous() or not vec.is_contiguous()", "!(x.isContiguous && y.isContiguous)"),
    ("inp.is_neg() or vec.is_neg()", "x.isNeg || y.isNeg"),
    ("N >= 2**63 or M >= 2**63", "rows x >= 2^63 || columns x >= 2^63"),
    ("(inp.numel() > 0 and inp.data_ptr() % 4 != 0) or (vec.numel() > 0 and vec.data_ptr() % 4 != 0)",
     "(x.numel > 0 && x.dataPtr.toNat % 4 != 0) || (y.numel > 0 && y.dataPtr.toNat % 4 != 0)"),
]

def check_source():
    original = function(ORIGINAL, "mv_kernel")
    candidate = function(SOURCE, "mv_contiguous_kernel")
    assert list(map(dump, original.decorator_list)) == list(map(dump, candidate.decorator_list))
    for i, arg in enumerate(["N", "M", "stride_an", "stride_am", "stride_bm", "stride_cn"]):
        assert dump(candidate.body[i]) == dump(statement(f"{arg} = tl.cast({arg}, tl.uint64)"))
    for i, expected in enumerate([
        "pid = tl.program_id(0).to(tl.uint64)",
        "offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.uint64)[:, None]",
        "offset_m = tl.arange(0, BLOCK_M).to(tl.uint64)[None, :]",
    ], 6):
        assert dump(candidate.body[i]) == dump(statement(expected))
    body = copy.deepcopy(candidate.body[6:])
    body[:3] = copy.deepcopy(original.body[:3])
    k = next(i for i, n in enumerate(body) if isinstance(n, ast.While))
    assert dump(body[k-1]) == dump(statement("m = tl.full((), 0, tl.uint64)"))
    assert dump(body[k].test) == dump(ast.parse("m < M", mode="eval").body)
    assert dump(body[k].body[-1]) == dump(statement("m += BLOCK_M"))
    loop = next(n for n in original.body if isinstance(n, ast.For))
    assert list(map(dump, body[k].body[:-1])) == list(map(dump, loop.body))
    body[k-1:k+1] = [copy.deepcopy(loop)]
    candidate.body, candidate.name = body, original.name
    assert dump(candidate) == dump(original), "unreviewed kernel change"

    wrapper = function(SOURCE, "mv")
    guards = [n.test for n in wrapper.body if isinstance(n, ast.If)]
    assert list(map(dump, guards)) == [dump(ast.parse(c, mode="eval").body) for c, _ in CHECKS]
    lean = " ".join((HERE / "Contract.lean").read_text().split())
    tail = lean.split("def checkTensorInputs", 1)[1].split("def checkInputs", 1)[0]
    assert re.findall(r"\bif (.*?) then", tail) == [c for _, c in CHECKS if c]
    assert "| _, _ => .error .tensors" in lean
    assert sum(isinstance(n, ast.Raise) for n in ast.walk(wrapper)) == len(CHECKS)
    launch = next(n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Subscript)
                  and isinstance(n.func.value, ast.Name) and n.func.value.id == "mv_contiguous_kernel")
    assert [dump(n) for n in launch.args] == [dump(ast.parse(s, mode="eval").body)
                                            for s in ["inp", "vec", "out", "N", "M", "M", "1", "1", "1"]]
    ordinary = [n for n in wrapper.body if not isinstance(n, ast.If)]
    expected = ast.parse("""N, M = inp.shape
out = torch.empty((N,), device=inp.device, dtype=inp.dtype)
grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
with torch.cuda.device(inp.device):
    mv_contiguous_kernel[grid](inp, vec, out, N, M, M, 1, 1, 1)
return out
""").body
    assert list(map(dump, ordinary)) == list(map(dump, expected))
    forbidden = {"clone", "contiguous", "resolve_neg", "reshape", "flatten"}
    assert not any(isinstance(n, ast.Attribute) and n.attr in forbidden for n in ast.walk(wrapper))
    pending, seen = [HERE / "Kernels.lean"], set()
    while pending:
        p = pending.pop()
        if p in seen: continue
        seen.add(p)
        for module in re.findall(r"^import (\S+)", p.read_text(), re.M):
            assert ".Proofs." not in module
            if module.startswith("bench."):
                pending.append(HERE.parents[3] / (module.replace(".", "/") + ".lean"))
    assert "Proofs.FP" not in (HERE / "Correct.lean").read_text()
    assert "Proofs.Real" not in (HERE / "FPEquiv.lean").read_text()

class TensorFacade:
    """Actual CPU storage and metadata, with explicitly mocked CUDA ownership."""
    def __init__(self, data, **overrides):
        self.data = data
        self.device = torch.device("cuda:0")
        self.is_cuda = True
        self.__dict__.update(overrides)
    @property
    def layout(self): return self.data.layout
    @property
    def dtype(self): return self.data.dtype
    @property
    def shape(self): return getattr(self, "shape_override", self.data.shape)
    @property
    def ndim(self): return len(self.shape)
    def numel(self): return self.data.numel()
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self):
        assert self.numel() > 0, "empty input pointer queried"
        return self.data.data_ptr()

class Tuner:
    def __init__(self, fn, configs):
        self.fn, self.configs, self.selected = fn, configs, configs[0]
        self.launches = 0
    def __getitem__(self, grid):
        def launch(*args):
            self.launches += 1
            args = tuple(a.data if isinstance(a, TensorFacade) else a for a in args)
            return self.fn[grid](*args, **self.selected.kwargs)
        return launch

old_tune = triton.autotune
triton.autotune = lambda configs, key: lambda fn: Tuner(fn, configs)
try:
    spec = importlib.util.spec_from_file_location("matrix_vector_candidate", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
finally:
    triton.autotune = old_tune

allocations = []
def empty(shape, device, dtype):
    assert device == torch.device("cuda:0")
    out = torch.empty(shape, dtype=dtype)
    allocations.append(out)
    return out
module.torch = SimpleNamespace(Tensor=TensorFacade, strided=torch.strided, float32=torch.float32,
                               empty=empty, cuda=SimpleNamespace(device=lambda _: nullcontext()))

def raw(t): return ctypes.string_at(t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
def wrap(t, **kwargs): return TensorFacade(t, **kwargs)

def unaligned(remainder, shape):
    count = 1
    for n in shape: count *= n
    backing = torch.empty(4*count+4, dtype=torch.uint8)
    storage = backing.untyped_storage()[remainder:remainder+4*count]
    stride = (shape[1], 1) if len(shape) == 2 else (1,)
    return torch.empty(0, dtype=torch.float32).set_(storage, 0, shape, stride)

rejected = 0
def reject(a, b):
    global rejected
    before = len(allocations), module.mv_contiguous_kernel.launches
    try:
        module.mv(a, b)
    except (TypeError, ValueError):
        rejected += 1
    else:
        raise AssertionError("unsupported input admitted")
    assert before == (len(allocations), module.mv_contiguous_kernel.launches)

executed = 0
def check(a, b, configs=None):
    global executed
    saved_a, saved_b = raw(a), raw(b)
    expected = a @ b
    for config in configs or module.mv_contiguous_kernel.configs:
        module.mv_contiguous_kernel.selected = config
        out = module.mv(wrap(a), wrap(b))
        assert out is allocations[-1]
        assert out.shape == (a.shape[0],) and out.dtype == torch.float32 and out.stride() == (1,)
        torch.testing.assert_close(out, expected, rtol=0, atol=0, equal_nan=True)
        assert raw(a) == saved_a and raw(b) == saved_b
        executed += 1

def run():
    check_source()
    a, b = torch.ones((2, 3)), torch.ones(3)
    reject(object(), wrap(b)); reject(wrap(a), object())
    reject(wrap(a.to_sparse()), wrap(b))
    reject(wrap(a.double()), wrap(b)); reject(wrap(a), wrap(b.double()))
    reject(wrap(a, is_cuda=False), wrap(b)); reject(wrap(a), wrap(b, device=torch.device("cuda:1")))
    reject(wrap(a[0]), wrap(b)); reject(wrap(a), wrap(b.reshape(1, 3)))
    reject(wrap(a), wrap(torch.ones(4)))
    reject(wrap(torch.ones(2, 6)[:, ::2]), wrap(b))
    reject(wrap(a), wrap(torch.ones(6)[::2]))
    reject(wrap(torch._neg_view(a)), wrap(b)); reject(wrap(a), wrap(torch._neg_view(b)))
    reject(wrap(a, shape_override=(2**63, 3)), wrap(b))
    reject(wrap(a, shape_override=(2, 2**63)), wrap(b, shape_override=(2**63,)))
    for rem in [1, 2, 3]:
        reject(wrap(unaligned(rem, (2, 3))), wrap(b))
        reject(wrap(a), wrap(unaligned(rem, (3,))))
    for n, m in [(0, 0), (0, 3), (3, 0), (9, 3), (9, 33), (7, 129), (7, 259)]:
        check(((torch.arange(n*m).reshape(n, m) % 7)-3).float(), ((torch.arange(m) % 3)-1).float())
    backing = torch.arange(32, dtype=torch.float32) % 5
    check(backing[1:16].reshape(3, 5), backing[2:7])
    check(torch.as_strided(torch.tensor([2.]), (1, 1), (2**63-1, 2**63-1)),
          torch.as_strided(torch.tensor([3.]), (1,), (2**63-1,)))
    check(unaligned(1, (0, 3)), torch.ones(3), module.mv_contiguous_kernel.configs[:1])
    check(torch.tensor([[float("nan"), 2.], [float("inf"), 1.]]), torch.ones(2),
          module.mv_contiguous_kernel.configs[:1])
    # Concrete original signed-i32 row*stride defect; contiguous N=3, M=2**30.
    old_offset = ctypes.c_int32(2 * 2**30).value
    new_offset = ctypes.c_uint64(2 * 2**30).value
    assert old_offset == -2**31 and new_offset == 2**31
    # Unresolved negative views change logical values without changing pointers.
    neg = torch._neg_view(torch.tensor([[2.]]))
    assert neg.data_ptr() == neg._base.data_ptr() and (neg @ torch.ones(1)).item() == -2
    print(f"PASS: {rejected} pre-allocation rejections; {executed} interpreted launches; "
          f"{len(module.mv_contiguous_kernel.configs)} source configurations; source/guard bindings; "
          "input storage frame, fresh return, empty/tail/alias/offset/NaN/Inf and counterexamples. No GPU run.")

if __name__ == "__main__": run()

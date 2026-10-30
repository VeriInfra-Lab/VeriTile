"""Focused CPU actual-source checks; this does not validate GPU arithmetic."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGINAL = ROOT / "bench/tritonbench_g/masked_add_cuda/masked_add_cuda.py"
REPAIRED = HERE / "masked_add_cuda_repaired.py"


class Array(np.ndarray):
    def to(self, dtype):
        return np.asarray(self, dtype=dtype).view(Array)


class Tensor:
    """Real CPU backing storage with explicit mocked CUDA metadata."""
    def __init__(self, value, **metadata):
        self.value = value
        self.metadata = metadata
        self.pointer_queries = 0

    def __getattr__(self, name):
        if name in self.metadata:
            return self.metadata[name]
        if name == "is_cuda":
            return True
        if name == "device":
            return "cuda:0"
        return getattr(self.value, name)

    def data_ptr(self):
        self.pointer_queries += 1
        if self.metadata.get("forbid_pointer"):
            raise AssertionError("empty tensor pointer was queried")
        return self.metadata.get("pointer", self.value.data_ptr())

    def numel(self):
        return self.metadata.get("numel", self.value.numel())

    def is_contiguous(self):
        return self.metadata.get("contiguous", self.value.is_contiguous())

    def is_neg(self):
        return self.metadata.get("negative", self.value.is_neg())

    def __add__(self, offsets):
        return self, np.asarray(offsets)

    def raw(self):
        count = self.value.untyped_storage().nbytes() // self.value.element_size()
        start = self.value.storage_offset()
        return torch.empty(0, dtype=self.value.dtype).set_(
            self.value.untyped_storage(), start, (count-start,), (1,)).numpy()


class Language:
    int1 = np.bool_
    constexpr = object()

    def __init__(self):
        self.pid = 0

    def program_id(self, axis):
        assert axis == 0
        return self.pid

    def arange(self, start, stop):
        return np.arange(start, stop).view(Array)

    def load(self, pointer, mask):
        tensor, offsets = pointer
        raw = tensor.raw()
        active = offsets[np.asarray(mask)]
        if np.any(active < 0) or np.any(active >= len(raw)):
            raise IndexError("source load outside backing storage")
        out = np.zeros(offsets.shape, dtype=raw.dtype).view(Array)
        out[np.asarray(mask)] = raw[active]
        return out

    def store(self, pointer, value, mask):
        tensor, offsets = pointer
        raw = tensor.raw()
        active = offsets[np.asarray(mask)]
        if np.any(active < 0) or np.any(active >= len(raw)):
            raise IndexError("source store outside backing storage")
        raw[active] = np.asarray(value)[np.asarray(mask)]


class JIT:
    def __init__(self, function, language, launches):
        self.function, self.language, self.launches = function, language, launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            fusion = kwargs.pop("enable_fp_fusion", None)
            extent = grid(kwargs)[0]
            self.launches.append((extent, kwargs["BLOCK_SIZE"], fusion, args[4]))
            for pid in range(extent):
                self.language.pid = pid
                with np.errstate(invalid="ignore", over="ignore"):
                    self.function(*args, **kwargs)
        return launch


def load_source(path):
    module = ast.parse(path.read_text())
    language, launches = Language(), []
    triton = SimpleNamespace(cdiv=lambda n, b: (n+b-1)//b)
    triton.jit = lambda f: JIT(f, language, launches)
    @contextmanager
    def device_context(device):
        assert device == "cuda:0"
        yield

    torch_api = SimpleNamespace(Tensor=Tensor, strided=torch.strided,
                                float32=torch.float32, int32=torch.int32, int64=torch.int64,
                                cuda=SimpleNamespace(device=device_context))
    namespace = {"torch": torch_api, "triton": triton, "tl": language}
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name in ("masked_add_kernel", "masked_add")]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["masked_add"], launches, functions[0]


def operands(n=5, mask_dtype=torch.int64):
    return (Tensor(torch.arange(n, dtype=torch.float32)),
            Tensor(torch.full((n,), 2.0)),
            Tensor(torch.zeros(n, dtype=mask_dtype)))


def main():
    repaired, launches, repaired_ast = load_source(REPAIRED)
    original, _, original_ast = load_source(ORIGINAL)
    assert ast.dump(original_ast) == ast.dump(repaired_ast), "kernel body changed"
    accepted = rejected = counterexamples = 0

    for dtype in (torch.int32, torch.int64):
        for n in (0, 1, 7, 1023, 1024, 1025, 2051):
            g, p, m = operands(n, dtype)
            if n:
                m.value[:] = torch.tensor([0, 1, -1, 2, -2] * ((n+4)//5), dtype=dtype)[:n]
                m.value[-1] = torch.iinfo(dtype).min
            old = g.value.clone()
            expected = torch.where(m.value == 0, old + p.value*0.5, old)
            initial_calls = len(launches)
            assert repaired(g, p, m, 0.5) is None
            assert torch.equal(g.value, expected)
            assert len(launches)-initial_calls == (n > 0)
            if n:
                assert launches[-1] == ((n+1023)//1024, 1024, True, 0.5)
            accepted += 1

    # Scalar, contiguous offset, unusual singleton strides and read-only aliases.
    for shape in ((), (1, 1), (2, 3)):
        n = int(np.prod(shape))
        g, p, m = operands(n)
        g.value = g.value.reshape(shape)
        p.value = p.value.reshape(shape)
        m.value = m.value.reshape(shape)
        assert repaired(g, p, m) is None
        accepted += 1
    g = Tensor(torch.arange(8, dtype=torch.float32)[1:6])
    p = Tensor(torch.ones(8, dtype=torch.float32)[2:7])
    m = Tensor(torch.zeros(8, dtype=torch.int64)[1:6])
    assert repaired(g, p, m, 1.0) is None
    assert torch.equal(g.value, torch.arange(2, 7, dtype=torch.float32))
    accepted += 1
    g, p, m = operands(1, torch.int32)
    g.value = g.value.as_strided((1, 1), (99, 7))
    p.value = p.value.as_strided((1, 1), (13, 2))
    m.value = m.value.as_strided((1, 1), (41, 9))
    assert repaired(g, p, m, 0.0) is None
    accepted += 1
    g, p, _ = operands(3, torch.int32)
    p.value.zero_()
    m = Tensor(p.value.view(torch.int32))
    assert repaired(g, p, m, 1.0) is None
    accepted += 1

    # Partially overlapping readonly views use the same bytes at both mask widths.
    for dtype in (torch.int32, torch.int64):
        backing = torch.zeros(14, dtype=torch.float32)
        data_view = backing[4:9]
        data_view.copy_(torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0]))
        mask_view = backing.view(dtype)[:5]
        grad = torch.arange(5, dtype=torch.float32)
        expected = torch.where(mask_view == 0, grad + data_view*0.5, grad)
        original_bytes = backing.view(torch.uint8).clone()
        assert repaired(Tensor(grad), Tensor(data_view), Tensor(mask_view), 0.5) is None
        assert torch.equal(grad, expected)
        assert torch.equal(backing.view(torch.uint8), original_bytes)
        accepted += 1

    # Empty inputs skip alignment and alias queries; exceptional alpha is admitted.
    for alpha in (0, -0.0, float("inf"), float("nan")):
        g, p, m = operands(0)
        for t in (g, p, m):
            t.metadata.update(forbid_pointer=True, pointer=1)
        assert repaired(g, p, m, alpha) is None
        assert all(t.pointer_queries == 0 for t in (g, p, m))
        accepted += 1
    g, p, m = operands(3)
    g.value[:] = torch.tensor([float("nan"), float("inf"), -0.0])
    m.value.fill_(1)
    before_bits = g.value.view(torch.int32).clone()
    assert repaired(g, p, m, float("inf")) is None
    assert torch.equal(g.value.view(torch.int32), before_bits)
    accepted += 1

    def reject(args, alpha=0.5):
        nonlocal rejected
        count = len(launches)
        try:
            repaired(*args, alpha)
        except (TypeError, ValueError):
            rejected += 1
        else:
            raise AssertionError("unsupported input admitted")
        assert len(launches) == count

    reject((None, *operands()[1:]))
    for slot, override in ((0, {"layout": torch.sparse_coo}),
                           (0, {"dtype": torch.float16}), (1, {"dtype": torch.float64}),
                           (2, {"dtype": torch.bool}), (2, {"dtype": torch.float32}),
                           (0, {"is_cuda": False}), (1, {"device": "cuda:1"}),
                           (2, {"shape": (4,)}), (0, {"contiguous": False}),
                           (1, {"negative": True}), (2, {"negative": True}),
                           (0, {"numel": 2**31})):
        args = operands()
        args[slot].metadata.update(override)
        reject(args)
    for alpha in (True, 1, np.float32(0.5), "0.5"):
        reject(operands(), alpha)
    for slot in range(3):
        args = operands()
        args[slot].metadata["pointer"] = args[slot].value.data_ptr()+1
        reject(args)
    for slot, dtype, width in ((0, torch.float32, 4), (1, torch.float32, 4),
                               (2, torch.int32, 4), (2, torch.int64, 8)):
        for remainder in range(1, width):
            raw = torch.zeros(128, dtype=torch.uint8)
            storage = raw.untyped_storage()[remainder:remainder+5*width]
            value = torch.empty(0, dtype=dtype).set_(storage, 0, (5,), (1,))
            assert value.is_contiguous() and value.data_ptr() % width == remainder
            args = list(operands(5, dtype if slot == 2 else torch.int64))
            args[slot] = Tensor(value)
            reject(args)
    args = operands()
    reject((args[0], args[0], args[2]))

    # Original stride-two addressing corrupts a storage hole and misses grad[1].
    gb = torch.tensor([10.0, 77.0, 20.0])
    pb = torch.tensor([2.0, 99.0, 4.0])
    mb = torch.zeros(3, dtype=torch.int64)
    args = (Tensor(gb[::2]), Tensor(pb[::2]), Tensor(mb[::2]))
    reject(args)
    original(*args, 0.5)
    assert gb.tolist() == [11.0, 126.5, 20.0]
    counterexamples += 1

    # Same strides do not establish equal extents: original issues an OOB load.
    args = (Tensor(torch.zeros(2)), Tensor(torch.ones(1)), Tensor(torch.zeros(2, dtype=torch.int64)))
    reject(args)
    try:
        original(*args, 1.0)
    except IndexError:
        counterexamples += 1
    else:
        raise AssertionError("shape counterexample did not execute OOB load")

    # Shifted aliases produce concrete cross-program dependencies.
    backing = torch.ones(1026)
    args = (Tensor(backing[1:]), Tensor(backing[:-1]), Tensor(torch.zeros(1025, dtype=torch.int64)))
    reject(args)
    original(*args, 1.0)
    assert backing[-1].item() == 3.0  # initial-snapshot formula is 2.0
    counterexamples += 1
    backing = torch.zeros(1026)
    args = (Tensor(backing[1:]), Tensor(torch.ones(1025)), Tensor(backing[:-1].view(torch.int32)))
    reject(args)
    original(*args, 1.0)
    assert backing[-1].item() == 0.0  # initial mask is zero: target is 1.0
    counterexamples += 1

    # Actual unresolved negative views are contiguous but direct loads disagree.
    g, p, m = operands(2)
    g.value = torch._neg_view(torch.tensor([10.0, 20.0]))
    reject((g, p, m))
    original(g, p, m, 0.5)
    assert g.value.tolist() == [-11.0, -21.0]  # logical target is [-9,-19]
    counterexamples += 1

    # Nonbinary mask contradicts the original docstring's (1-mask) expression.
    g, p, m = operands(1)
    g.value.fill_(10)
    m.value.fill_(2)
    original(g, p, m, 0.5)
    assert g.value.item() == 10.0 and 10+2*(1-2)*0.5 == 9.0
    counterexamples += 1

    from triton.runtime.jit import mangle_type
    assert mangle_type(2**31) == "i64"
    assert mangle_type(2**31+1) == "i64"
    wrapped = np.array([2**31], dtype=np.uint32).view(np.int32)[0]
    assert int(wrapped) == -(2**31) and int(wrapped) < 2**31+1
    counterexamples += 1

    # Public layers stay independent; source file contains no proof declarations.
    assert "FPEquiv" not in (HERE / "Correct.lean").read_text()
    assert "Correct" not in (HERE / "FPEquiv.lean").read_text()
    kernel_text = (HERE / "Kernels.lean").read_text()
    assert "theorem " not in kernel_text and ".Proofs." not in kernel_text
    print(f"passed: {accepted} admissions, {rejected} rejections, {counterexamples} original counterexamples")
    print("kernel AST unchanged; CPU backing/interpreted source only; no GPU numerical validation")


if __name__ == "__main__":
    main()

"""Execute the actual wrapper and JIT AST on CPU storage; no GPU claim."""
import ast
import contextlib
import copy
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "layernorm_fwd_triton_repaired.py"
spec = importlib.util.spec_from_file_location("fused_source_adapter", HERE.parent / "fused_layernorm_triton/check_source.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Tensor:
    def __init__(self, value, cuda=True, device=0):
        self.value = value
        self.device = torch.device("cuda" if cuda else "cpu", device)

    def __getattr__(self, name):
        return getattr(self.value, name)


class TL(base.TL):
    sum = staticmethod(lambda x, axis: np.sum(x, axis=axis, dtype=np.float32))
    sqrt = staticmethod(np.sqrt)

    def program_id(self, axis):
        return np.int64(self.ids[axis])

    @staticmethod
    def load(ptr, mask, other=0):
        offsets = np.asarray(ptr.offset)
        mask = np.broadcast_to(mask, offsets.shape)
        indexes = offsets[mask]
        assert np.all((indexes >= 0) & (indexes < len(ptr.array)))
        value = np.full(offsets.shape, other, dtype=np.float32)
        value[mask] = ptr.array[indexes]
        return value


class Casts(base.Casts):
    def visit_Attribute(self, node):
        if node.attr == "element_ty":
            return ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="float32", ctx=ast.Load())
        return self.generic_visit(node)


class Launch:
    def __init__(self, fn, tl, records, strict):
        self.fn, self.tl, self.records, self.strict = fn, tl, records, strict

    def __getitem__(self, grid):
        def call(*args):
            bound = inspect.signature(self.fn).bind(*args)
            tensors, writes = {}, {}
            for name, value in list(bound.arguments.items()):
                if isinstance(value, Tensor):
                    raw = value.value.untyped_storage()
                    backing = torch.empty(0, dtype=torch.float32).set_(raw, 0, (raw.nbytes() // 4,), (1,))
                    array = backing.detach().numpy()
                    counter = np.zeros(array.size, dtype=np.int64)
                    tensors[name], writes[name] = value.value, counter
                    bound.arguments[name] = base.Pointer(array, value.storage_offset(), writes=counter)
                elif name == "eps":
                    bound.arguments[name] = np.float32(value)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for seq in range(grid[0]):
                    for head in range(grid[1]):
                        self.tl.ids = seq, head
                        self.fn(*bound.args)
            expected = np.zeros_like(writes["Y"])
            y = tensors["Y"]
            for coord in np.ndindex(tuple(y.shape)):
                index = y.storage_offset() + sum(i * s for i, s in zip(coord, y.stride()))
                expected[index] += 1
            if self.strict:
                assert np.array_equal(writes["Y"], expected), (grid, y.shape, writes["Y"], expected)
            assert not writes["X"].any() and not writes["W"].any()
            self.records.append((grid, writes["Y"].copy(), expected))
        return call


def environment(source=SOURCE, strict=True):
    tl, records = TL(), []
    ns = {"math": math, "tl": tl, "torch": SimpleNamespace(
        Tensor=Tensor, float32=torch.float32, strided=torch.strided,
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()),
        empty_like=lambda t: Tensor(torch.empty_like(t.value), device=t.device.index))}
    for node in ast.parse(source.read_text()).body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("test_"):
            continue
        node = copy.deepcopy(node)
        if node.name == "_layer_norm_fwd_kernel":
            node.decorator_list = []
            node = Casts().visit(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), ns)
        if node.name == "_layer_norm_fwd_kernel":
            ns[node.name] = Launch(ns[node.name], tl, records, strict)
    return ns, records


def main():
    torch.manual_seed(29)
    ns, records = environment()
    run = ns["layernorm_forward"]
    cases = []
    for shape in ((0, 2, 3), (2, 0, 3), (2, 3, 0), (1, 1, 1),
                  (1, 2, 2), (2, 3, 127), (2, 2, 128), (2, 3, 129), (1, 2, 259)):
        s, h, n = shape
        x = torch.randn(s, n, h).transpose(1, 2)
        w = torch.randn(n, h).t()
        cases.append((x, w))
    # Offset, stride-zero and aliased X/W inputs are read directly.
    cases.append((torch.randn(3, 2, 130)[1:, :, 1:130:2], torch.randn(2, 130)[:, 1:130:2]))
    cases.append((torch.randn(1, 1, 5).expand(2, 3, 5), torch.randn(1, 5).expand(3, 5)))
    shared = torch.randn(2, 3, 5)
    cases.append((shared, shared[0]))
    singleton = torch.as_strided(torch.ones(1), (1, 1, 1), (2**62, 2**62, 2**62))
    cases.append((singleton, singleton[0]))
    for x, w in cases:
        x0, w0 = x.clone(), w.clone()
        y = run(Tensor(x), Tensor(w), 1e-5).value
        if x.numel():
            ref = (x - x.mean(-1, keepdim=True)) * torch.rsqrt(x.var(-1, unbiased=False, keepdim=True) + 1e-5) * w
            torch.testing.assert_close(y, ref, rtol=7e-5, atol=3e-6)
        assert y.shape == x.shape and y.dtype == torch.float32
        assert torch.equal(x, x0) and torch.equal(w, w0)
    # Actual original source reproduces colliding writes on inner stride two.
    old, old_records = environment(HERE.parents[2] / "tritonbench_g/layernorm_fwd_triton/layernorm_fwd_triton.py", strict=False)
    x = torch.tensor([[[1., 3.], [2., 4.]]]).transpose(1, 2)
    w = torch.ones(2, 2)
    old["layernorm_forward"](Tensor(x), Tensor(w), 1e-5)
    assert np.array_equal(old_records[-1][1], [1, 2, 1, 0])
    run(Tensor(x), Tensor(w), 1e-5)
    rejects = [(x, Tensor(w), 1e-5), (Tensor(x.double()), Tensor(w), 1e-5),
               (Tensor(x, cuda=False), Tensor(w), 1e-5),
               (Tensor(torch._neg_view(x)), Tensor(w), 1e-5),
               (Tensor(x), Tensor(w, device=1), 1e-5),
               (Tensor(x[0]), Tensor(w), 1e-5), (Tensor(x), Tensor(w[:1]), 1e-5),
               *[(Tensor(x), Tensor(w), eps) for eps in (0, -1, float("inf"), float("nan"), "bad", 10**1000)]]
    for byte in (1, 2, 3):
        raw = torch.empty(16 + byte, dtype=torch.uint8).untyped_storage()[byte:]
        misaligned = torch.empty(0, dtype=torch.float32).set_(raw, 0, (2, 2), (2, 1))
        rejects.append((Tensor(x), Tensor(misaligned), 1e-5))
    for args in rejects:
        try:
            run(*args)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("unsupported input admitted")
    run(Tensor(x), Tensor(w), True)
    print(f"PASS: {len(cases)} source wrapper/kernel cases, complete stores, original inputs unchanged")
    print(f"PASS: {len(rejects)} guard rejections, bool epsilon, original inner-stride collision")
    print("CPU storage/source adapter; no native GPU numerical conformance claimed.")


if __name__ == "__main__":
    main()

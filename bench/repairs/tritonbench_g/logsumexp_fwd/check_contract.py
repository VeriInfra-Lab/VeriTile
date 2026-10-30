"""Actual wrapper/JIT body on bounded NumPy storage and CPU Torch library calls.

This checks source behavior and frames, not native GPU numerical conformance.
"""
import ast
import contextlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch as cpu_torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "logsumexp_fwd_repaired.py"
spec = importlib.util.spec_from_file_location("lse_test_memory", HERE.parent / "apply_penalty/check_contract.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class Tensor(shared.Tensor):
    allocations = []
    views = 0
    reductions = 0
    casts = 0

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def view(self, *shape):
        Tensor.views += 1
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        assert self.is_contiguous()
        return Tensor(self.data.reshape(shape), device=self.device.name)

    def new_empty(self, *shape, dtype):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        count = math.prod(shape)
        backing = np.full(count+8, 61, dtype=dtype)
        result = Tensor(backing[4:4+count].reshape(shape), device=self.device.name)
        Tensor.allocations.append(backing)
        return result

    def new_full(self, shape, value, dtype):
        result = self.new_empty(shape, dtype=dtype)
        result.data.fill(value)
        return result

    def logsumexp(self, dim):
        Tensor.reductions += 1
        result = cpu_torch.logsumexp(cpu_torch.from_numpy(self.data), dim).numpy()
        return Tensor(result, device=self.device.name)

    def to(self, dtype):
        Tensor.casts += 1
        return Tensor(self.data.astype(dtype), device=self.device.name)


class TL(shared.TL):
    tensor = object()
    float32 = np.float32
    pids = (0, 0)
    max = staticmethod(lambda values, dim: np.fmax.reduce(values, axis=dim))
    sum = staticmethod(lambda values, dim: np.sum(values, axis=dim, dtype=np.float32))
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    cdiv = staticmethod(lambda n, d: (n+d-1)//d)

    @classmethod
    def program_id(cls, dim):
        return cls.pids[dim]


def load_wrapper():
    source = ast.parse(SOURCE.read_text())
    kernel = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name.endswith("_kernel"))
    kernel.decorator_list = []
    kernel = shared.EraseIndexCasts().visit(kernel)
    env = {"tl": TL, "math": math, "Optional": Optional}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[])), str(SOURCE), "exec"), env)
    body = env[kernel.name]
    calls = []
    # Every retained warp configuration may run as a tuning probe before the
    # final launch. All must read input and overwrite the same scratch cells.
    profiles = [1, 2, 4, 8, 16, 32, 4]

    class Kernel:
        def __getitem__(self, grid):
            def launch(**kwargs):
                for warps in profiles:
                    calls.append((grid, warps, kwargs["B"]))
                    # Disjoint scratch slots permit any complete program order.
                    for p in reversed(list(np.ndindex(grid))):
                        TL.pids = p
                        args = {k: shared.Pointer(v) if isinstance(v, Tensor) else v for k, v in kwargs.items()}
                        args["scale"] = None if kwargs["scale"] is None else np.float32(kwargs["scale"])
                        args["HAS_SCALE"] = kwargs["scale"] is not None
                        body(**args)
                # Check the actual kernel outputs before the retained Torch
                # row aggregation. This oracle is independent of max shifting.
                raw = kwargs["x"].data
                scaled = raw if kwargs["scale"] is None else raw*np.float32(kwargs["scale"])
                for block in range(grid[1]):
                    part = scaled[:, block*kwargs["B"]:min((block+1)*kwargs["B"], kwargs["D"])]
                    expected = cpu_torch.logsumexp(cpu_torch.from_numpy(part), -1).numpy()
                    np.testing.assert_allclose(kwargs["z"].data[:, block], expected,
                                               rtol=5e-6, atol=5e-6, equal_nan=True)
            return launch

    torch = SimpleNamespace(Tensor=Tensor, strided="strided", float32=np.dtype("float32"),
        float64=np.dtype("float64"), dtype=np.dtype,
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()))
    env.update(torch=torch, triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length(), cdiv=TL.cdiv),
               logsumexp_fwd_kernel=Kernel())
    wrapper = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "logsumexp_fwd")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(SOURCE), "exec"), env)
    return env[wrapper.name], calls


def main():
    wrapper, calls = load_wrapper()
    rng = np.random.default_rng(129)
    checks = 0
    cases = []
    for shape in ((1,), (3,), (3, 17), (2, 3, 65), (1, 65536), (1, 65537), (1, 131073),
                  (0,), (4, 0), (2, 0, 0), (0, 7), (2, 0, 65)):
        data = rng.normal(size=shape).astype(np.float32)
        for scale in (None, 0.0, -0.5, 2.0):
            for dtype in (None, np.dtype("float32"), np.dtype("float64")):
                cases.append((data, scale, dtype))
    for row in ((-np.inf, -np.inf, -np.inf), (np.inf, 2, -3), (np.nan, 2, -np.inf),
                (-0.0, 0.0, 1e-40), (80, -80, 0)):
        for scale in (None, 0.0, -2.0, np.inf, -np.inf, np.nan):
            cases.append((np.asarray([row], np.float32), scale, None))
    with np.errstate(all="ignore"):
        for data, scale, dtype in cases:
            # The Tensor starts inside a backing allocation, so data_ptr and
            # logical index zero must respect a nonzero storage offset.
            backing = np.full(data.size+8, 71, np.float32)
            backing[4:4+data.size] = data.reshape(-1)
            x = Tensor(backing[4:4+data.size].reshape(data.shape))
            before = x.data.copy()
            before_calls = len(calls)
            out = wrapper(x, scale, dtype)
            scaled = before if scale is None else before*np.float32(scale)
            expected = cpu_torch.logsumexp(cpu_torch.from_numpy(scaled), -1).numpy()
            if dtype is not None:
                expected = expected.astype(dtype)
            np.testing.assert_allclose(out.data, expected, rtol=5e-6, atol=5e-6, equal_nan=True)
            assert out.shape == data.shape[:-1] and out.dtype == (np.dtype("float32") if dtype is None else dtype)
            assert out.device == x.device
            np.testing.assert_array_equal(x.data, before)
            assert np.all(backing[:4] == 71) and np.all(backing[-4:] == 71)
            if data.size == 0:
                assert len(calls) == before_calls
            checks += 1
    invalid = [([1], None, None), (Tensor([1], np.float64), None, None),
               (Tensor([1], np.float32, device="cpu:0"), None, None),
               (Tensor(1, np.float32), None, None),
               (Tensor(np.ones((2, 3), np.float32).T), None, None),
               (Tensor([1], np.float32, negative=True), None, None),
               (Tensor([1], np.float32), 1, None), (Tensor([1], np.float32), True, None),
               (Tensor([1], np.float32), Tensor([1], np.float32), None),
               (Tensor([1], np.float32), None, np.dtype("float16"))]
    misaligned = Tensor([1], np.float32)
    misaligned.pointer_override = 1
    invalid.append((misaligned, None, None))
    extent = Tensor([1], np.float32)
    extent.size_override = 2**31
    invalid.append((extent, None, None))
    output_extent = Tensor(np.empty((0,), np.float32))
    output_extent.shape = (2**31, 0)
    output_extent.ndim = 2
    invalid.append((output_extent, None, None))
    for args in invalid:
        previous = (len(calls), len(Tensor.allocations), Tensor.views)
        try:
            wrapper(*args)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(args)
        assert previous == (len(calls), len(Tensor.allocations), Tensor.views)
        checks += 1
    for backing in Tensor.allocations:
        assert np.all(backing[:4] == 61) and np.all(backing[-4:] == 61)
    assert Tensor.casts and Tensor.reductions and Tensor.views
    assert {warps for _, warps, _ in calls} == {1, 2, 4, 8, 16, 32}
    print(f"logsumexp_fwd: {checks} actual-wrapper/body cases passed; no GPU execution")
    print("Per-block kernel oracle, reversed program order, nonzero storage offsets and input/output sentinels passed.")


if __name__ == "__main__":
    main()

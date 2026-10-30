"""Replay the original one-launch wrapper over CPU storage; no GPU claim."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "matrix_reduction.py"


def functions():
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in ("load_reduce_kernel", "load_reduce")]
    assert len(nodes) == 2
    for node in nodes:
        node.decorator_list = []
        for arg in node.args.args:
            arg.annotation = None
    return nodes


class Pointer:
    def __init__(self, tensor, offsets=None):
        self.tensor, self.offsets = tensor, offsets

    def __add__(self, offsets):
        return Pointer(self.tensor, offsets)


def run():
    calls = []
    allocated = []

    def make_block_ptr(*, base, shape, strides, offsets, block_shape, order):
        assert shape == block_shape and offsets == (0, 0) and order == (1, 0)
        assert strides == base.tensor.stride()
        return base

    def store(pointer, value):
        pointer.tensor[pointer.offsets] = value.to(pointer.tensor.dtype)

    def randn(shape, device, dtype):
        assert device == "cuda"
        calls.append(("randn", shape, dtype))
        # Non-NaN inputs include signed zeros, infinities and exact subnormals.
        values = torch.tensor([-0.0, 0.0, -float("inf"), float("inf"), 1e-40, -1.0, 2.0], dtype=dtype)
        return values.repeat((shape[0]*shape[1]+6)//7)[:shape[0]*shape[1]].reshape(shape)

    def empty(shape, device, dtype):
        assert device == "cuda"
        calls.append(("empty", shape, dtype))
        backing = torch.full((shape[0]+4,), -37.0, dtype=dtype)
        allocated.append(backing)
        return backing[2:shape[0]+2]

    def compare(actual, expected, **kwargs):
        assert kwargs == dict(rtol=1e-2, atol=1e-3, check_dtype=False)
        calls.append(("assert", tuple(actual.shape), actual.dtype))
        torch.testing.assert_close(actual, expected, **kwargs)

    namespace = {"tl": SimpleNamespace(make_block_ptr=make_block_ptr, load=lambda ptr: ptr.tensor.clone(),
                                       max=lambda value, axis: value.to(torch.float32).max(dim=axis).values,
                                       arange=torch.arange, store=store),
                 "torch": SimpleNamespace(float16=torch.float16, float32=torch.float32,
                                          randn=randn, empty=empty,
                                          set_printoptions=lambda **kw: calls.append(("print", kw))),
                 "assert_close": compare}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions(), type_ignores=[])), str(SOURCE), "exec"), namespace)
    kernel = namespace["load_reduce_kernel"]

    class Launch:
        def __getitem__(self, grid):
            assert grid == (1,)
            def execute(x, y, sm, sn, sy, m, n):
                calls.append(("launch", grid, sm, sn, sy, m, n))
                kernel(Pointer(x), Pointer(y), sm, sn, sy, m, n)
            return execute

    namespace["load_reduce_kernel"] = Launch()
    count = 0
    for dtype in ("float16", "float32"):
        for m, n in ((1, 1), (4, 8), (16, 16), (32, 32), (64, 64), (128, 128)):
            calls.clear()
            assert namespace["load_reduce"](m, n, dtype) is None
            backing = allocated[-1]
            assert torch.equal(backing[:2], torch.full_like(backing[:2], -37.0))
            assert torch.equal(backing[-2:], torch.full_like(backing[-2:], -37.0))
            assert [call[0] for call in calls] == ["randn", "empty", "launch", "print", "assert"]
            count += 1
    calls.clear()
    try:
        namespace["load_reduce"](4, 8, "bfloat16")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown dtype must fail at the original mapping")
    assert calls == []
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()
    assert "FPEquiv" not in (HERE / "Correct.lean").read_text()
    assert "import bench.tritonbench_g.matrix_reduction.Correct" not in (HERE / "FPEquiv.lean").read_text()
    print(f"Passed {count} original wrapper/kernel CPU replays, dtype rejection, exact launch, destination exterior sentinels and None return.")
    print("Both source dtype profiles retained; CUDA launch mocked, no GPU numerical validation.")


if __name__ == "__main__":
    run()

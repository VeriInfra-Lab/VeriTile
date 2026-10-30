"""Exercise actual wrapper/JIT bodies with bounded NumPy storage; no GPU claim."""
import ast
import contextlib
import copy
import importlib.util
import inspect
import math
import operator
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "max_reduction_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/max_reduction/max_reduction.py"
spec = importlib.util.spec_from_file_location("max_test_memory", HERE.parent / "apply_penalty/check_contract.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class Tensor(shared.Tensor):
    contiguous_calls = 0

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def contiguous(self):
        Tensor.contiguous_calls += 1
        if self.is_contiguous():
            return self
        return Tensor(np.ascontiguousarray(self.data), device=self.device.name)


class TL(shared.TL):
    tensor = object()
    int32 = np.int32
    pids = (0, 0, 0)

    @classmethod
    def program_id(cls, axis):
        return cls.pids[axis]

    @staticmethod
    def max(values, axis=None, return_indices=False):
        result = np.max(values, axis=axis)
        if return_indices:
            return result, np.argmax(values, axis=axis).astype(np.int32)
        return result

    @classmethod
    def store(cls, ptr, value, mask=True):
        offsets, masks, values = np.broadcast_arrays(ptr.offset, mask, value)
        assert np.all(offsets[masks] >= 0)
        assert np.all(offsets[masks] < ptr.tensor.data.size)
        ptr.tensor.data.reshape(-1)[offsets[masks]] = values[masks]


class ArrayBooleans(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        values = [self.visit(v) for v in node.values]
        result = values[0]
        for value in values[1:]:
            result = ast.BinOp(result, ast.BitAnd() if isinstance(node.op, ast.And) else ast.BitOr(), value)
        return result


def load_wrapper():
    original = ast.parse(ORIGINAL.read_text())
    source = ast.parse(SOURCE.read_text())
    env = {"tl": TL, "math": math, "operator": operator, "namedtuple": namedtuple}
    allocations, calls = [], []
    metadata = {"BLOCK_M": 8}

    class Kernel:
        def __init__(self, body):
            self.body = body

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                selected = dict(metadata)
                bound = inspect.signature(self.body).bind_partial(*args, **kwargs)
                if self.body.__name__ == "max_kernel":
                    selected["BLOCK_N"] = 1 << (bound.arguments["N"] - 1).bit_length()
                    kwargs.update(selected)
                actual_grid = grid(selected) if callable(grid) else grid
                calls.append((self.body.__name__, actual_grid, selected))
                counts = tuple(actual_grid) + (1,) * (3-len(actual_grid))
                for p in np.ndindex(counts):
                    TL.pids = p
                    self.body(*(shared.Pointer(x) if isinstance(x, Tensor) else x for x in args), **kwargs)
            return launch

    names = ("max_kernel_1", "max_kernel_2", "max_kernel")
    for name in names:
        baseline = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == name)
        candidate = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
        baseline, candidate = copy.deepcopy(baseline), copy.deepcopy(candidate)
        baseline.decorator_list = candidate.decorator_list = []
        candidate = shared.EraseIndexCasts().visit(candidate)
        assert ast.dump(baseline, include_attributes=False) == ast.dump(candidate, include_attributes=False), name
        executable = ArrayBooleans().visit(candidate)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[executable], type_ignores=[])), str(SOURCE), "exec"), env)
        env[name] = Kernel(env[name])

    def empty(shape, dtype, device):
        shape = tuple(shape)
        count = math.prod(shape)
        backing = np.full(count+8, 61, dtype=dtype)
        tensor = Tensor(backing[4:4+count].reshape(shape), device=device.name)
        allocations.append((tensor, backing))
        return tensor

    def squeeze(tensor, dim):
        return Tensor(np.squeeze(tensor.data, axis=dim), device=tensor.device.name)

    env.update(torch=SimpleNamespace(Tensor=Tensor, strided="strided", float32=np.dtype("float32"),
        int64=np.dtype("int64"), empty=empty, squeeze=squeeze,
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext())),
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length(), cdiv=lambda n, d: (n+d-1)//d))
    wrappers = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name not in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=wrappers, type_ignores=[])), str(SOURCE), "exec"), env)
    return env, allocations, calls, metadata


def main():
    env, allocations, calls, metadata = load_wrapper()
    checks = 0
    rng = np.random.default_rng(17)
    for count in (1, 2, 3, 15, 16, 17, 255, 256, 257, 1025, 4097):
        x = Tensor(rng.normal(size=count).astype(np.float32))
        before = x.data.copy()
        result = env["max"](x)
        assert result.shape == () and result.dtype == np.float32 and result.device == x.device
        assert result.data == np.max(before)
        np.testing.assert_array_equal(x.data, before)
        checks += 1
    for bm in (8, 16, 32):
        metadata["BLOCK_M"] = bm
        for shape, axis in (((1,), 0), ((17, 13, 3), 1), ((3, 5, 7), -1),
                            ((7, 3, 5), 0), ((0, 7, 3), 1), ((3, 7, 0), 1)):
            for keep in (False, True):
                data = rng.integers(-7, 8, size=shape).astype(np.float32)
                x = Tensor(data)
                before = x.data.copy()
                before_calls = len(calls)
                result = env["max_dim"](x, axis, keep)
                np.testing.assert_array_equal(result.values.data, np.max(data, axis=axis, keepdims=keep))
                expected_index = np.argmax(data, axis=axis)
                if keep:
                    expected_index = np.expand_dims(expected_index, axis)
                np.testing.assert_array_equal(result.indices.data, expected_index)
                assert result.indices.dtype == np.int64 and result.values.dtype == np.float32
                assert result.values.device == x.device and result.indices.device == x.device
                np.testing.assert_array_equal(x.data, before)
                if data.size == 0:
                    assert len(calls) == before_calls
                checks += 1
        x = Tensor(np.arange(105, dtype=np.float32).reshape(3, 5, 7).transpose(2, 0, 1))
        before = x.data.copy()
        result = env["max_dim"](x, -2, True)
        np.testing.assert_array_equal(result.values.data, np.max(before, axis=-2, keepdims=True))
        np.testing.assert_array_equal(x.data, before)
        checks += 1
    for keep in (0, 1, [], [1]):
        x = Tensor(np.asarray([[2, 7, 7]], dtype=np.float32))
        result = env["max_dim"](x, 1, keep)
        assert result.values.shape == ((1, 1) if keep else (1,))
        assert result.indices.data.reshape(-1)[0] == 1
        checks += 1
    for shape in ((), (1,), (1, 1)):
        x = Tensor(np.ones(shape, dtype=np.float32))
        assert env["max"](x).data == 1
        checks += 1
    rejected = [("max", [1], ()), ("max", Tensor([], np.float32), ()),
        ("max", Tensor([1], np.float64), ()), ("max", Tensor([1], np.float32, device="cpu:0"), ()),
        ("max", Tensor([1], np.float32, negative=True), ()),
        ("max", Tensor(np.ones((2, 3), np.float32).T), ()),
        ("max_dim", Tensor([1], np.float32), (None,)),
        ("max_dim", Tensor([1], np.float32), (1,)),
        ("max_dim", Tensor([1], np.float32), (-2,)),
        ("max_dim", Tensor(np.ones((1, 0), np.float32)), (1,)),
        ("max_dim", Tensor(np.ones((1, 32769), np.float32)), (1,)),
        ("max_dim", Tensor(np.ones((1, 1, 65536), np.float32)), (1,))]
    extent = Tensor([1], np.float32)
    extent.size_override = 2**31
    rejected.append(("max", extent, ()))
    misaligned = Tensor([1], np.float32)
    misaligned.pointer_override = 1
    rejected.append(("max", misaligned, ()))
    for wrapper, x, args in rejected:
        before_calls, before_allocs = len(calls), len(allocations)
        before = x.data.copy() if isinstance(x, Tensor) else None
        try:
            env[wrapper](x, *args)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError((wrapper, args))
        assert len(calls) == before_calls and len(allocations) == before_allocs
        if before is not None:
            np.testing.assert_array_equal(x.data, before)
        checks += 1
    for _, backing in allocations:
        assert np.all(backing[:4] == 61) and np.all(backing[-4:] == 61)
    assert Tensor.contiguous_calls > 0
    for boundary in range(1, 46342):
        for n in (boundary*boundary-1, boundary*boundary, boundary*boundary+1):
            if 0 < n < 2**31:
                assert math.isqrt(n-1)+1 == math.ceil(math.sqrt(n))
    print(f"max_reduction: {checks} actual-wrapper/body cases and sqrt boundaries passed; no GPU execution")


if __name__ == "__main__":
    main()

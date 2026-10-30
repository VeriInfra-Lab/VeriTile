"""Execute both actual wrappers and stripped JIT bodies on bounded FP32 storage.

CUDA is mocked; this checks source arithmetic and whole-call behavior, not FMA
or GPU transcendental conformance. CPU Torch supplies independent references.
"""
import ast
import contextlib
import importlib.util
import inspect
import math
import operator
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch as cpu_torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from bench.jit_correspondence import check_manifest
SOURCE = HERE / "log_softmax_repaired.py"
spec = importlib.util.spec_from_file_location("softmax_test_memory", HERE.parent / "max_reduction/check_contract.py")
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


class Tensor(shared.Tensor):
    pass


class TL(shared.TL):
    float32 = np.float32
    minimum = staticmethod(np.minimum)
    max = staticmethod(lambda values, axis: np.fmax.reduce(values, axis=axis))
    sum = staticmethod(lambda values, axis: np.sum(values, axis=axis, dtype=np.float32))
    exp = staticmethod(np.exp)
    log = staticmethod(np.log)


class EraseCasts(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            assert len(node.args) == 1 and ast.unparse(node.args[0]) in ("tl.int64", "tl.float32")
            return node.func.value
        if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == "tl.cast":
            assert len(node.args) == 2 and ast.unparse(node.args[1]) == "tl.int64"
            return node.args[0]
        return node


def load_wrapper():
    source = ast.parse(SOURCE.read_text())
    env = {"tl": TL, "math": math, "operator": operator, "logging": SimpleNamespace(debug=lambda _: None)}
    calls, allocations = [], []
    block = {"BLOCK_M": 1}

    class Kernel:
        def __init__(self, body):
            self.body = body

        def __getitem__(self, grid):
            def launch(*args, **options):
                assert options == {"enable_fp_fusion": True}
                bound = inspect.signature(self.body).bind_partial(*args)
                n = bound.arguments["N"]
                meta = dict(block, BLOCK_N=1 << (n-1).bit_length())
                actual_grid = grid(meta)
                calls.append((self.body.__name__, actual_grid, meta.copy()))
                for p in np.ndindex(actual_grid):
                    TL.pids = p
                    with np.errstate(all="ignore"):
                        self.body(*(shared.shared.Pointer(x) if isinstance(x, shared.Tensor) else x for x in args), **meta)
            return launch

    kernels = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name.endswith("kernel_repaired")]
    for kernel in kernels:
        kernel.decorator_list = []
        stripped = EraseCasts().visit(kernel)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[stripped], type_ignores=[])), str(SOURCE), "exec"), env)
        env[kernel.name] = Kernel(env[kernel.name])

    def empty_like(x, dtype=None):
        dtype = x.dtype if dtype is None else dtype
        backing = np.full(x.numel()+8, 61, dtype=dtype)
        allocations.append(backing)
        return Tensor(backing[4:4+x.numel()].reshape(x.shape), device=x.device.name)

    class Context:
        def save_for_backward(self, *tensors):
            self.saved_tensors = tensors

    class Function:
        @classmethod
        def apply(cls, *args):
            ctx = Context()
            result = cls.forward(ctx, *args)
            result.context = ctx
            return result

    env.update(torch=SimpleNamespace(Tensor=shared.Tensor, strided="strided", float32=np.dtype("float32"),
        empty_like=empty_like, autograd=SimpleNamespace(Function=Function),
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext())),
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length(), cdiv=lambda n, b: (n+b-1)//b))
    definitions = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n not in kernels]
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])), str(SOURCE), "exec"), env)
    return env, block, calls, allocations


def main():
    assert len(check_manifest(ROOT, HERE / "jit-correspondence.json")) == 4
    env, block, calls, allocations = load_wrapper()
    wrapper = env["log_softmax_repaired"]
    backward = env["LogSoftmaxRepaired"].backward
    rng = np.random.default_rng(523)
    checks = 0
    with np.errstate(all="ignore"):
        for bm in (1, 2, 4, 8):
            block["BLOCK_M"] = bm
            for shape, dim in (((1,), 0), ((17, 31, 3), 1), ((3, 5, 7), -1), ((3, 5, 7), 0),
                              ((3, 1025), 1), ((2, 2049), -1), ((0, 7, 3), 1), ((3, 0, 2), 1), ((3, 7, 0), 1), ((0, 1048577), 1), ((0, 1, 65536), 1)):
                x = Tensor(rng.normal(size=shape).astype(np.float32))
                data = x.data.copy()
                before_calls = len(calls)
                out = wrapper(x, dim)
                ref = cpu_torch.log_softmax(cpu_torch.from_numpy(data), dim).numpy()
                np.testing.assert_allclose(out.data, ref, rtol=2e-5, atol=2e-5)
                assert out.shape == x.shape and out.dtype == np.float32 and out.device == x.device
                assert out.context.saved_tensors == (out,) and out.context.dim == dim % len(shape)
                # Reverse view tests the original backward contiguous call.
                grad = Tensor(rng.normal(size=shape).astype(np.float32)[..., ::-1])
                grad_before = grad.data.copy()
                y_before = out.data.copy()
                dx, ddim, ddtype = backward(out.context, grad)
                expected = grad_before-np.exp(y_before)*np.sum(grad_before, axis=dim, keepdims=True)
                np.testing.assert_allclose(dx.data, expected, rtol=2e-5, atol=2e-5)
                assert ddim is None and ddtype is None and dx.shape == x.shape and dx.dtype == np.float32
                np.testing.assert_array_equal(x.data, data)
                np.testing.assert_array_equal(out.data, y_before)
                np.testing.assert_array_equal(grad.data, grad_before)
                if data.size == 0:
                    assert len(calls) == before_calls
                checks += 1
            for data in (np.asarray([[0, -1000, -10000]], np.float32),
                         np.asarray([[np.inf, 0, -np.inf]], np.float32),
                         np.asarray([[-np.inf, -np.inf, -np.inf]], np.float32),
                         np.asarray([[np.nan, 0, 3]], np.float32)):
                out = wrapper(Tensor(data), -1, np.dtype("float32"))
                reference = cpu_torch.log_softmax(cpu_torch.from_numpy(data), -1).numpy()
                np.testing.assert_allclose(out.data, reference, rtol=2e-5, atol=2e-5, equal_nan=True)
                checks += 1
            data = rng.normal(size=(3, 5, 7)).astype(np.float32).transpose(2, 0, 1)
            out = wrapper(Tensor(data), -2)
            np.testing.assert_allclose(out.data, cpu_torch.log_softmax(cpu_torch.from_numpy(data), -2).numpy(),
                                       rtol=2e-5, atol=2e-5)
            checks += 1
    rejects = [([1], -1, None), (Tensor(1, np.float32), 0, None), (Tensor([1], np.float64), 0, None),
               (Tensor([1], np.float32, device="cpu:0"), 0, None),
               (Tensor([1], np.float32, negative=True), 0, None), (Tensor([1], np.float32), 1, None),
               (Tensor([1], np.float32), -2, None), (Tensor([1], np.float32), 0.0, None),
               (Tensor([1], np.float32), 0, cpu_torch.float64),
               (Tensor(np.ones((1, 131073), np.float32)), 1, None),
               (Tensor(np.ones((1, 1, 65536), np.float32)), 1, None)]
    misaligned = Tensor([1], np.float32)
    misaligned.pointer_override = 1
    rejects.append((misaligned, 0, None))
    extent = Tensor([1], np.float32)
    extent.size_override = 2**31
    rejects.append((extent, 0, None))
    for args in rejects:
        before = len(calls), len(allocations), shared.Tensor.contiguous_calls
        try:
            wrapper(*args)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(args)
        assert before == (len(calls), len(allocations), shared.Tensor.contiguous_calls)
        checks += 1
    out = wrapper(Tensor(np.ones((2, 3), np.float32)), 1)
    for grad in (Tensor(np.ones((3, 2), np.float32)), Tensor(np.ones((2, 3), np.float32), device="cuda:1"),
                 Tensor(np.ones((2, 3), np.float64)), Tensor(np.ones((2, 3), np.float32), negative=True)):
        before = len(calls), len(allocations), shared.Tensor.contiguous_calls
        try:
            backward(out.context, grad)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("invalid gradient admitted")
        assert before == (len(calls), len(allocations), shared.Tensor.contiguous_calls)
        checks += 1
    assert shared.Tensor.contiguous_calls > 0
    for backing in allocations:
        assert np.all(backing[:4] == 61) and np.all(backing[-4:] == 61)
    print(f"log_softmax: {checks} actual forward/backward wrapper/body cases passed; no GPU execution")


if __name__ == "__main__":
    main()

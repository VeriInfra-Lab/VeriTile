"""Actual wrapper/autograd/JIT AST execution on CPU storage, with CUDA metadata adapted."""
import ast
import contextlib
import copy
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import triton

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fast_layernorm_repaired.py"
spec = importlib.util.spec_from_file_location("layernorm_source_adapter", HERE.parent / "layernorm_fwd_triton/check_source.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Tensor(base.Tensor):
    def __getattr__(self, name):
        result = getattr(self.value, name)
        if name in ("contiguous", "view", "reshape"):
            return lambda *args, **kwargs: Tensor(result(*args, **kwargs), device=self.device.index)
        return result


class Function:
    @classmethod
    def apply(cls, *args):
        ctx = SimpleNamespace()
        ctx.save_for_backward = lambda *xs: setattr(ctx, "saved_tensors", xs)
        cls.last_context = ctx
        return cls.forward(ctx, *args)


class TL(base.TL):
    math = SimpleNamespace(rsqrt=lambda x: np.float32(1) / np.sqrt(x))

    def program_id(self, axis):
        assert axis == 0
        return np.int64(self.row)

    @staticmethod
    def load(ptr, mask=None, other=0):
        offsets = np.asarray(ptr.offset)
        if mask is None:
            mask = np.ones(offsets.shape, dtype=bool)
        return base.TL.load(ptr, mask, other)

    @staticmethod
    def store(ptr, value, mask=None):
        return base.base.TL.store(ptr, value, mask)


class Launch:
    def __init__(self, fn, tl, records):
        self.fn, self.tl, self.records = fn, tl, records

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            warps = kwargs.pop("num_warps", 4)
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            writes = {}
            for name, value in list(bound.arguments.items()):
                if isinstance(value, Tensor):
                    raw = value.untyped_storage()
                    array = torch.empty(0, dtype=torch.float32).set_(raw, 0, (raw.nbytes() // 4,), (1,)).numpy()
                    counter = np.zeros(array.size, dtype=np.int64)
                    writes[name] = counter
                    bound.arguments[name] = base.base.Pointer(array, value.storage_offset(), writes=counter)
                elif name == "eps":
                    bound.arguments[name] = np.float32(value)
            for row in range(grid[0]):
                self.tl.row = row
                self.fn(*bound.args, **bound.kwargs)
            if self.fn.__name__ == "layernorm_forward":
                expected = ("Y", "r", "mu")
            elif self.fn.__name__ == "layernorm_backward":
                expected = ("dX",)
            else:
                expected = ("dW", "db") if bound.arguments["HAS_BIAS"] else ("dW",)
            for name, counter in writes.items():
                if name in expected:
                    assert np.all(counter == 1), (self.fn.__name__, name, counter)
                else:
                    assert not counter.any(), (self.fn.__name__, name)
            self.records.append((self.fn.__name__, grid, warps))
        return call


def environment(source=SOURCE):
    tl, records = TL(), []
    api = SimpleNamespace(Tensor=Tensor, float32=torch.float32, strided=torch.strided,
        autograd=SimpleNamespace(Function=Function), cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()),
        empty=lambda *args, dtype, device: Tensor(torch.empty(*args, dtype=dtype), device=torch.device(device).index or 0),
        empty_like=lambda t: Tensor(torch.empty_like(t.value), device=t.device.index))
    ns = {"torch": api, "tl": tl, "triton": triton, "math": math}
    for node in ast.parse(source.read_text().split("################################################################")[0]).body:
        if isinstance(node, ast.Import) or isinstance(node, ast.Expr):
            continue
        node = copy.deepcopy(node)
        if isinstance(node, ast.FunctionDef) and node.name in ("layernorm_forward", "layernorm_backward", "layernorm_affine_backward"):
            node.decorator_list = []
            node = base.Casts().visit(node)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), ns)
            ns[node.name] = Launch(ns[node.name], tl, records)
        else:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    return ns, records


def main():
    torch.manual_seed(41)
    ns, records = environment()
    cases = [(0, 3, False), (1, 1, True), (3, 3, True), (2, 127, False), (2, 128, True),
             (2, 129, True), (2, 2048, True), (1, 8192, False), (1, 32768, True), (1, 65536, False)]
    for rows, width, bias in cases:
        x = torch.randn(rows * width + 1)[1:].view(rows, width)
        w = torch.randn(width + 4)[1:]
        b = torch.randn(width + 7)[2:].view(1, width + 5) if bias else None
        if rows == 1:
            x = torch.as_strided(x, (rows, width), (2**62, 1))
        incoming = torch.randn(1).expand_as(x)
        incoming_stride = incoming.stride()
        x0, w0 = x.clone(), w.clone()
        b0 = b.clone() if b is not None else None
        module = SimpleNamespace(elementwise_affine=True, weight=Tensor(w), bias=Tensor(b) if b is not None else None,
                                 eps=9.0, variance_epsilon=1e-5)
        output = ns["fast_layernorm"](module, Tensor(x))
        fn = ns["Fast_Layernorm"]
        ctx = fn.last_context
        assert ctx.eps == 1e-5
        assert ctx.saved_tensors[0].data_ptr() == x.data_ptr()
        assert ctx.saved_tensors[1].data_ptr() == w.data_ptr()
        result = fn.backward(ctx, Tensor(incoming))
        xr, wr = x.detach().requires_grad_(), w.detach().requires_grad_()
        br = b.detach().requires_grad_() if b is not None else None
        reference = torch.nn.functional.layer_norm(xr, (width,), None, None, 1e-5) * wr.reshape(-1)[:width]
        if br is not None:
            reference = reference + br.reshape(-1)[:width]
        targets = torch.autograd.grad(reference, (xr, wr, br) if br is not None else (xr, wr), incoming)
        torch.testing.assert_close(output.value, reference, rtol=6e-5, atol=4e-6)
        for actual, expected in zip(result, targets):
            torch.testing.assert_close(actual.value, expected, rtol=1e-3, atol=1e-4)
        assert len(result) == 4 and result[3] is None and (result[2] is None) == (b is None)
        assert torch.equal(x, x0) and torch.equal(w, w0)
        assert b is None or torch.equal(b, b0)
        assert incoming.stride() == incoming_stride
        assert torch.count_nonzero(result[1].value[width:]) == 0
        if b is not None:
            assert torch.count_nonzero(result[2].value.reshape(-1)[width:]) == 0
    check = ns["_checked_tensor"]
    rejects = [(torch.ones(3),), (Tensor(torch.ones(3).double()),), (Tensor(torch.ones(3), cuda=False),),
               (Tensor(torch.ones(3), device=1), torch.device("cuda", 0)),
               (Tensor(torch.randn(3, 2).t()),), (Tensor(torch._neg_view(torch.ones(3))),)]
    for byte in (1, 2, 3):
        raw = torch.empty(12 + byte, dtype=torch.uint8).untyped_storage()[byte:]
        bad = torch.empty(0, dtype=torch.float32).set_(raw, 0, (3,), (1,))
        rejects.append((Tensor(bad),))
    for args in rejects:
        try:
            check(*args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("unsupported Tensor admitted")
    bad_calls = [(torch.ones(()), torch.ones(3), None, 1e-5),
                 (torch.empty(2, 0), torch.ones(3), None, 1e-5),
                 (torch.ones(1, 65537), torch.ones(65537), None, 1e-5),
                 (torch.ones(1, 3), torch.ones(2), None, 1e-5),
                 (torch.ones(1, 3), torch.ones(3), torch.ones(2), 1e-5),
                 *[(torch.ones(1, 3), torch.ones(3), None, e) for e in (0, -1, float("inf"), float("nan"), "bad", 10**1000)]]
    for x, w, b, eps in bad_calls:
        try:
            ns["Fast_Layernorm"].apply(Tensor(x), Tensor(w), Tensor(b) if b is not None else None, eps)
        except (ValueError, TypeError, RuntimeError):
            pass
        else:
            raise AssertionError("unsupported forward request admitted")
    old, _ = environment(HERE.parents[2] / "tritonbench_g/fast_layernorm/fast_layernorm.py")
    x, w, b = torch.tensor([[2.25, 3., 3.75]]), torch.ones(3), torch.zeros(3)
    fixed = ns["Fast_Layernorm"].apply(Tensor(x), Tensor(w), Tensor(b), 0.625).value
    broken = old["Fast_Layernorm"].apply(Tensor(x), Tensor(w), Tensor(b), 0.625).value
    torch.testing.assert_close(fixed, torch.tensor([[-0.75, 0., 0.75]]), rtol=0, atol=0)
    torch.testing.assert_close(broken, torch.tensor([[-0.375, 0., 0.375]]), rtol=0, atol=0)
    for width, warps in ((1, 4), (2048, 8), (8192, 16), (32768, 32), (65536, 32)):
        assert ns["calculate_settings"](width)[1] == warps
    print(f"PASS: {len(cases)} complete source forward/autograd/backward cases, offset/singleton inputs, expanded dY, optional bias, affine tails and all warp settings")
    print(f"PASS: {len(rejects) + len(bad_calls)} guard rejections, original inputs preserved and original padding counterexample")
    print("CPU source/storage adapter; no GPU numerical conformance claimed.")


if __name__ == "__main__":
    main()

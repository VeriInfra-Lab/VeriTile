"""Execute the actual wrapper and both JIT ASTs on CPU storage; no GPU claim."""
import ast
import contextlib
import copy
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "rbe_triton_transform_repaired.py"
spec = importlib.util.spec_from_file_location("rbe_cpu_base", HERE.parent / "layernorm_fwd_triton/check_source.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
Tensor = base.Tensor


class TL(base.TL):
    cdiv = staticmethod(lambda x, y: (x + y - 1) // y)
    cos, sin = staticmethod(np.cos), staticmethod(np.sin)
    extra = SimpleNamespace(cuda=SimpleNamespace(libdevice=SimpleNamespace(
        pow=lambda x, y: np.power(np.float32(x), y))))

    @staticmethod
    def divide(x, y):
        return np.asarray(x, dtype=np.float32) / np.asarray(y, dtype=np.float32)


class Casts(base.Casts):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Attribute(value=ast.Name(id="tl", ctx=ast.Load()), attr="divide", ctx=ast.Load()),
                            args=[node.left, node.right], keywords=[])
        return node


class Launch:
    def __init__(self, fn, tl, records):
        self.fn, self.tl, self.records = fn, tl, records

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            bound = inspect.signature(self.fn).bind(*args, **kwargs)
            shape = grid(bound.arguments) if callable(grid) else grid
            writes = {}
            for name, value in list(bound.arguments.items()):
                if isinstance(value, Tensor):
                    raw = value.untyped_storage()
                    array = torch.empty(0, dtype=torch.float32).set_(raw, 0, (raw.nbytes() // 4,), (1,)).numpy()
                    counter = np.zeros(array.size, dtype=np.int64)
                    writes[name] = counter
                    bound.arguments[name] = base.base.Pointer(array, value.storage_offset(), writes=counter)
            for batch in range(shape[0]):
                for program in range(shape[1]):
                    self.tl.ids = batch, program
                    self.fn(*bound.args, **bound.kwargs)
            assert not writes["x_ptr"].any()
            assert np.all(writes["out_ptr"] == 1)
            self.records.append(shape)
        return call


def environment():
    tl, records = TL(), []
    ns = {"tl": tl, "triton": SimpleNamespace(cdiv=TL.cdiv), "torch": SimpleNamespace(
        Tensor=Tensor, float32=torch.float32, strided=torch.strided, contiguous_format=torch.contiguous_format,
        cuda=SimpleNamespace(device=lambda _: contextlib.nullcontext()),
        empty_like=lambda t, **kw: Tensor(torch.empty_like(t.value, **kw), device=t.device.index))}
    for node in ast.parse(SOURCE.read_text()).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        node = copy.deepcopy(node)
        if node.decorator_list:
            node.decorator_list = []
            node = Casts().visit(node)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), ns)
        if node.name == "rbe_triton":
            ns[node.name] = Launch(ns[node.name], tl, records)
    return ns, records


def reference(x, pos):
    x = torch.empty_like(x, memory_format=torch.contiguous_format).copy_(x)
    pairs = torch.arange(0, x.shape[-1], 2)
    denominator = torch.pow(torch.tensor(10000., dtype=torch.float32), (pairs % 128).float() / 128)
    positions = (torch.arange(x.shape[1], dtype=torch.int64) + pos).float()
    angle = positions[:, None] / denominator[None, :]
    cosine, sine = angle.cos(), angle.sin()
    real = x[..., ::2]
    imag = torch.zeros_like(real)
    imag[..., :x.shape[-1] // 2] = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., ::2] = real * cosine - imag * sine
    out[..., 1::2] = (real * sine + imag * cosine)[..., :x.shape[-1] // 2]
    return out


def main():
    torch.manual_seed(29)
    ns, records = environment()
    run = ns["rbe_triton_wrapper"]
    cases = []
    for shape in ((0, 2, 3), (2, 0, 3), (2, 3, 0), (1, 1, 1), (2, 3, 3),
                  (2, 3, 127), (1, 2, 128), (2, 3, 1023), (2, 3, 1024),
                  (1, 3, 1025), (1, 5, 2051)):
        raw = torch.randn(int(np.prod(shape)) + 7)
        x = raw[3:3 + int(np.prod(shape))].view(shape)
        cases.append((x, -3))
    cases.extend([(torch.as_strided(torch.ones(1), (1, 1, 1), (2**62, 2**62, 2**62)), True),
                  (torch.randn(2, 3, 7), 2**33), (torch.randn(1, 1, 3), -(2**63)),
                  (torch.randn(1, 1, 3), 2**63 - 2)])
    for x, pos in cases:
        raw = torch.empty(0, dtype=torch.uint8).set_(x.untyped_storage(), 0, (x.untyped_storage().nbytes(),), (1,))
        before = raw.clone()
        y = run(Tensor(x), pos).value
        assert torch.equal(raw, before)
        assert y.shape == x.shape and y.dtype == torch.float32 and y.is_contiguous()
        assert y.numel() == 0 or y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        if x.numel():
            torch.testing.assert_close(y, reference(x, pos), rtol=5e-5, atol=4e-6)
    x = torch.randn(2, 3, 4)
    rejects = [(x, 0), (Tensor(x.double()), 0), (Tensor(x, cuda=False), 0),
               (Tensor(x[0]), 0), (Tensor(x.transpose(1, 2)), 0), (Tensor(torch._neg_view(x)), 0),
               *[(Tensor(x), p) for p in (1.5, "0", 2**63, -(2**63)-1, 2**63-3)]]
    for byte in (1, 2, 3):
        raw = torch.empty(4 * 24 + byte, dtype=torch.uint8).untyped_storage()[byte:]
        bad = torch.empty(0, dtype=torch.float32).set_(raw, 0, (2, 3, 4), (12, 4, 1))
        rejects.append((Tensor(bad), 0))
    for args in rejects:
        try:
            run(*args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("unsupported request admitted")
    assert np.int64(65536) * np.int64(65536) == 2**32
    assert (65536 * 65536) % 2**32 == 0
    print(f"PASS: {len(cases)} actual-source cases, {len(rejects)} rejections, complete stores and full input-backing frame")
    print("CPU source/storage adapter; no native GPU numerical conformance claimed.")


if __name__ == "__main__":
    main()

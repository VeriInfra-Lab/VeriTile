"""Execute the actual wrapper/autograd source and JIT ASTs on CPU storage.
Only CUDA metadata/launch and Triton primitive operations are adapted. This is
not native GPU numerical validation.
"""
import ast
import copy
import functools
import importlib.util
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import triton

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "layer_norm_liger_repaired.py"

# Reuse the existing source-AST adapter; no second interpreter framework.
spec = importlib.util.spec_from_file_location("fused_source_adapter", HERE.parent / "fused_layernorm_triton/check_source.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


class Tensor:
    def __init__(self, value, cuda=True, device=0):
        self.value, self.is_cuda, self.device = value, cuda, torch.device("cuda", device)

    def __getattr__(self, name):
        result = getattr(self.value, name)
        if name in ("contiguous", "view", "sum", "to"):
            def call(*args, **kwargs):
                return Tensor(result(*args, **kwargs), self.is_cuda, self.device.index)
            return call
        return result


class TL(base.TL):
    sum = staticmethod(lambda x, axis: np.sum(x, axis=axis, dtype=np.float32))
    rsqrt = staticmethod(lambda x: np.float32(1) / np.sqrt(x))

    @staticmethod
    def load(ptr, mask=None, other=0, **kwargs):
        offsets = np.asarray(ptr.offset)
        mask = np.ones(offsets.shape, dtype=bool) if mask is None else np.broadcast_to(mask, offsets.shape)
        indexes = offsets[mask]
        assert np.all((indexes >= 0) & (indexes < len(ptr.array)))
        value = np.full(offsets.shape, other, dtype=np.float32)
        value[mask] = ptr.array[indexes]
        return value

    @staticmethod
    def store(ptr, values, mask=None):
        return base.TL.store(ptr, values, mask)


class Launch:
    def __init__(self, fn, tl, records, device_state):
        self.fn, self.tl, self.records, self.device_state = fn, tl, records, device_state

    def __getitem__(self, grid):
        def call(*args, **kwargs):
            assert all(value.device == self.device_state.current for value in args if isinstance(value, Tensor))
            self.device_state.launches.append((self.fn.__name__, self.device_state.current))
            if self.device_state.fail == self.fn.__name__:
                raise RuntimeError("mock submission failure")
            options = {k: kwargs.pop(k) for k in ("num_warps",) if k in kwargs}
            signature = inspect.signature(self.fn)
            bound = signature.bind(*args, **kwargs)
            writes = {}
            for name, value in list(bound.arguments.items()):
                if isinstance(value, Tensor):
                    assert value.value.is_contiguous()
                    array = value.value.detach().numpy().reshape(-1)
                    counter = np.zeros(array.size, dtype=np.int64)
                    writes[name] = counter
                    bound.arguments[name] = base.Pointer(array, writes=counter)
                elif name == "eps":
                    bound.arguments[name] = np.float32(value)
            for row in range(grid[0]):
                self.tl.row = row
                self.fn(*bound.args, **bound.kwargs)
            expected = ("Y_ptr", "Mean_ptr", "RSTD_ptr") if "forward" in self.fn.__name__ else ("DX_ptr", "DW_ptr", "DB_ptr")
            for name in expected:
                assert np.all(writes[name] == 1), (name, grid, writes[name])
            self.records.append((self.fn.__name__, grid, options, kwargs))
        return call


def environment(source=SOURCE):
    tree = ast.parse(source.read_text())
    tl, records = TL(), []
    sm = SimpleNamespace(multi_processor_count=4)
    device_state = SimpleNamespace(current=torch.device("cuda", 0), launches=[], fail=None)
    class DeviceContext:
        def __init__(self, device): self.device = device
        def __enter__(self):
            self.previous = device_state.current
            device_state.current = self.device
        def __exit__(self, *exc): device_state.current = self.previous
    torch_api = SimpleNamespace(
        Tensor=Tensor, float32=torch.float32, bfloat16=torch.bfloat16, strided=torch.strided,
        version=SimpleNamespace(hip=None), autograd=SimpleNamespace(Function=object),
        cuda=SimpleNamespace(get_device_properties=lambda device: sm, device=DeviceContext, state=device_state),
        empty=lambda *args, dtype, device: Tensor(torch.empty(*args, dtype=dtype), device=device.index))
    ns = {"torch": torch_api, "tl": tl, "triton": triton, "math": math, "functools": functools}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_layer_norm_"):
            kernel = copy.deepcopy(node)
            kernel.decorator_list = []
            kernel = base.Casts().visit(kernel)
            exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel], type_ignores=[])), str(source), "exec"), ns)
            ns[node.name] = Launch(ns[node.name], tl, records, device_state)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not getattr(node, "name", "").startswith("test_"):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), ns)
    return ns, sm, records


def main():
    torch.manual_seed(13)
    ns, sm, records = environment()
    function = ns["LigerLayerNormFunction"]
    device_state = ns["torch"].cuda.state
    cases = [(1, 1, 7), (1, 3, 7), (3, 5, 2), (5, 127, 4), (3, 128, 5),
             (2, 1025, 1), (2, 2048, 3), (2, 8192, 4), (1, 32768, 2), (1, 65536, 2)]
    for rows, width, programs in cases:
        sm.multi_processor_count = programs
        raw = torch.randn(rows, 2, width).transpose(0, 1)
        w = torch.randn(width, 2)[:, 0]
        b = torch.randn(width, 2)[:, 0]
        incoming = torch.randn(1).expand(raw.shape)
        incoming_stride = incoming.stride()
        x0, w0, b0 = raw.clone(), w.clone(), b.clone()
        ctx = SimpleNamespace()
        ctx.save_for_backward = lambda *xs: setattr(ctx, "saved_tensors", xs)
        device = 1 if (rows, width) == (1, 1) else 0
        out = function.forward(ctx, Tensor(raw, device=device), Tensor(w, device=device), Tensor(b, device=device), 1e-5)
        assert device_state.current == torch.device("cuda", 0)
        grad = function.backward(ctx, Tensor(incoming, device=device))
        assert device_state.current == torch.device("cuda", 0)
        assert all(d == torch.device("cuda", device) for _, d in device_state.launches[-2:])
        assert len(grad) == 4 and grad[3] is None
        reference_x, reference_w, reference_b = raw.detach().requires_grad_(), w.detach().requires_grad_(), b.detach().requires_grad_()
        reference = torch.nn.functional.layer_norm(reference_x, (width,), reference_w, reference_b, 1e-5)
        targets = torch.autograd.grad(reference, (reference_x, reference_w, reference_b), incoming)
        torch.testing.assert_close(out.value, reference, rtol=4e-5, atol=3e-6)
        for actual, expected in zip(grad, targets):
            torch.testing.assert_close(actual.value, expected, rtol=8e-4, atol=8e-5)
        assert grad[0].shape == raw.shape and grad[1].shape == w.shape and grad[2].shape == b.shape
        assert torch.equal(raw, x0) and torch.equal(w, w0) and torch.equal(b, b0)
        assert incoming.stride() == incoming_stride
    # Preserve the original forward-only HIP warp selection and backward default.
    for width, expected in [(1, 4), (2048, 8), (8192, 16), (32768, 32), (65536, 32)]:
        assert ns["calculate_settings"](width)[1] == expected
    ns["torch"].version.hip = "test"
    assert ns["calculate_settings"](32768)[1] == 16
    assert all(record[2] == {} for record in records if "backward" in record[0])
    ns["torch"].version.hip = None
    # Concrete padded-variance failure retained from the original source.
    x = torch.tensor([[2.25, 3.0, 3.75]])
    w, b = torch.ones(3), torch.zeros(3)
    fixed = ns["layer_norm_forward"](Tensor(x), Tensor(w), Tensor(b), 0.625)[0].value
    old, _, _ = environment(HERE.parents[2] / "tritonbench_g/layer_norm_liger/layer_norm_liger.py")
    broken = old["layer_norm_forward"](Tensor(x), Tensor(w), Tensor(b), 0.625)[0].value
    torch.testing.assert_close(fixed, torch.tensor([[-0.75, 0., 0.75]]), rtol=0, atol=0)
    torch.testing.assert_close(broken, torch.tensor([[-0.375, 0., 0.375]]), rtol=0, atol=0)
    check = ns["_checked_inputs"]
    rejects = [(x, Tensor(w), Tensor(b), 1e-5),
               (Tensor(x.double()), Tensor(w), Tensor(b), 1e-5),
               (Tensor(x, cuda=False), Tensor(w), Tensor(b), 1e-5),
               (Tensor(torch._neg_view(x)), Tensor(w), Tensor(b), 1e-5),
               (Tensor(x), Tensor(w, device=1), Tensor(b), 1e-5),
               (Tensor(torch.empty(0, 3)), Tensor(w), Tensor(b), 1e-5),
               (Tensor(x), Tensor(w[:2]), Tensor(b), 1e-5),
               *[(Tensor(x), Tensor(w), Tensor(b), eps) for eps in (0, -1, float("inf"), float("nan"), "bad")]]
    for byte in (1, 2, 3):
        storage = torch.empty(12 + byte, dtype=torch.uint8).untyped_storage()[byte:]
        bad = torch.empty(0, dtype=torch.float32).set_(storage, 0, (3,), (1,))
        rejects.append((Tensor(x), Tensor(bad), Tensor(b), 1e-5))
    for args in rejects:
        try:
            check(*args)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("unsupported input admitted")
    # Both actual submission sites restore an unrelated caller device on failure.
    owned_x, owned_w, owned_b = Tensor(x, device=1), Tensor(w, device=1), Tensor(b, device=1)
    ctx = SimpleNamespace()
    ctx.save_for_backward = lambda *xs: setattr(ctx, "saved_tensors", xs)
    function.forward(ctx, owned_x, owned_w, owned_b, 0.625)
    for kernel, call in [
        ("_layer_norm_forward_kernel", lambda: function.forward(ctx, owned_x, owned_w, owned_b, 0.625)),
        ("_layer_norm_backward_kernel", lambda: function.backward(ctx, Tensor(torch.ones_like(x), device=1))),
    ]:
        device_state.fail = kernel
        try:
            call()
        except RuntimeError as error:
            assert str(error) == "mock submission failure"
        else:
            raise AssertionError("expected mock submission failure")
        finally:
            device_state.fail = None
        assert device_state.current == torch.device("cuda", 0)
        assert device_state.launches[-1] == (kernel, torch.device("cuda", 1))
    print(f"PASS: {len(cases)} full forward/autograd/backward source executions, noncontiguous inputs, expanded dY, empty groups, all warp thresholds")
    print(f"PASS: {len(rejects)} guard rejections and exact original padding counterexample")
    print("PASS: owning-device forward/backward launches and caller restoration on success and both submission failures")
    print("CPU source tests; CUDA metadata and launch adapted; no GPU numerical conformance claimed.")


if __name__ == "__main__":
    main()

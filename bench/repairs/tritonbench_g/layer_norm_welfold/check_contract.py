"""Targeted source, admission and CPU execution checks; CUDA is mocked."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "layer_norm_welfold_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/layer_norm_welfold/layer_norm_welfold.py"
KERNEL = "triton_red_fused_native_layer_norm_no_welford"
WRAPPER = "fused_native_layer_norm_no_welford"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def source_checks():
    original = function(ORIGINAL, KERNEL)
    loads = [n for n in ast.walk(original) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "tl.load"]
    loads.sort(key=lambda n: n.lineno)
    for load in loads[:2]: load.keywords.insert(0, ast.keyword(arg="other", value=ast.Constant(0.0)))
    square = next(n for n in ast.walk(original) if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "tmp10")
    square.value = ast.parse("tl.where(rmask, tmp9 * tmp9, 0.0)", mode="eval").body
    assert ast.dump(original) == ast.dump(function(SOURCE, KERNEL)), "only the three tail fixes are permitted"
    old_host, new_host = function(ORIGINAL, WRAPPER), function(SOURCE, WRAPPER)
    assert ast.dump(ast.Module(body=old_host.body[1:], type_ignores=[])) == ast.dump(ast.Module(body=new_host.body[1:], type_ignores=[]))
    assert ast.unparse(new_host.body[0]) == "S, D = _check_inputs(primals_1, primals_2, primals_3)"
    guards = [ast.unparse(n.test) for n in function(SOURCE, "_check_inputs").body if isinstance(n, ast.If)]
    assert guards == [
        "not all((isinstance(t, torch.Tensor) for t in (weight, bias, x)))",
        "any((t.layout != torch.strided for t in (weight, bias, x)))",
        "any((t.dtype != torch.float32 for t in (weight, bias, x)))",
        "any((not t.is_cuda or t.device.index != 0 for t in (weight, bias, x)))",
        "x.ndim != 2", "D == 0", "weight.shape != (D,) or bias.shape != (D,)",
        "x.stride() != (D, 1) or weight.stride() != (1,) or bias.stride() != (1,)",
        "any((t.storage_offset() != 0 for t in (weight, bias, x)))",
        "any((t.is_neg() for t in (weight, bias, x)))",
        "D + 2048 >= 2 ** 31 or S * D + 2048 >= 2 ** 31",
        "S * D > 0 and x.data_ptr() % 16 != 0 or weight.data_ptr() % 16 != 0 or bias.data_ptr() % 16 != 0"]
    kernel_lean = (HERE / "Kernels.lean").read_text()
    assert ".Proofs." not in kernel_lean and "theorem " not in kernel_lean
    assert "guarded = tl.where(rmask, square, 0.0)" in kernel_lean
    assert "stored = (tmp21).to(tl.bfloat16)" in kernel_lean
    contract = (HERE / "Contract.lean").read_text()
    for required in ["0 < D", "[D,1]", "D + 2048 < 2^31", "S * D + 2048 < 2^31",
                     "dataPtr.toNat % 16 = 0", "isNeg = Bool.false"]:
        assert required in contract


class Tensor:
    def __init__(self, data, *, cuda=True, device=0, shape=None, strides=None):
        self.data, self.is_cuda = data, cuda
        self.device = SimpleNamespace(index=device)
        self._shape, self._strides = shape, strides
    @property
    def shape(self): return tuple(self.data.shape) if self._shape is None else self._shape
    @property
    def ndim(self): return len(self.shape)
    @property
    def layout(self): return self.data.layout
    @property
    def dtype(self): return self.data.dtype
    def stride(self): return self.data.stride() if self._strides is None else self._strides
    def storage_offset(self): return self.data.storage_offset()
    def is_neg(self): return self.data.is_neg()
    def data_ptr(self):
        assert self.data.numel() > 0, "empty input pointer must not be queried"
        return self.data.data_ptr()


def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))


class Pointer:
    def __init__(self, tensor, offsets=None): self.tensor, self.offsets = tensor, offsets
    def __add__(self, offsets): return Pointer(self.tensor, offsets)


def interpreter():
    pid, calls, allocations, current, selected = [0], [], [], [1], [1024]
    def load(pointer, mask, other=float("nan"), **kwargs):
        data, offsets = pointer.tensor.data, pointer.offsets
        values = torch.full(offsets.shape, other, dtype=data.dtype)
        values[mask] = raw(data)[data.storage_offset() + offsets[mask]]
        return values
    def store(pointer, value, mask):
        data, offsets = pointer.tensor.data, pointer.offsets
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        raw(data)[data.storage_offset() + offsets[mask]] = value.expand(offsets.shape)[mask].to(data.dtype)
    tl = SimpleNamespace(program_id=lambda axis: pid[0], arange=torch.arange,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        float32=torch.float32, load=load, store=store, broadcast_to=torch.broadcast_to,
        sum=lambda value, axis: value.sum(axis), where=torch.where, debug_barrier=lambda: None)
    namespace = {"tl": tl, "libdevice": SimpleNamespace(rsqrt=torch.rsqrt)}
    def compile_kernel(node):
        node = copy.deepcopy(node)
        node.decorator_list = []
        for arg in node.args.args: arg.annotation = None
        scope = dict(namespace)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), scope)
        return scope[KERNEL]
    kernel = compile_kernel(function(SOURCE, KERNEL))
    class Launch:
        def __getitem__(self, grid):
            def run(mean, inverse, x, w, bias, y, S, D):
                assert current[0] == 0
                assert grid({"XBLOCK": 1, "RBLOCK": selected[0]}) == (S,)
                calls.append("launch")
                for block in (1024, 2048, selected[0]):
                    for row in range(S):
                        pid[0] = row
                        kernel(*(Pointer(t) for t in (mean, inverse, x, w, bias, y)), S, D, 1, block)
            return run
    class DeviceGuard:
        def __init__(self, device): assert device == 0
        def __enter__(self):
            self.previous = current[0]
            current[0] = 0
            calls.append("guard")
        def __exit__(self, *exc): current[0] = self.previous
    def set_device(device):
        assert device == 0
        current[0] = device
        calls.append("device")
    def allocate(shape, strides, dtype):
        assert current[0] == 0
        calls.append("allocation")
        value = Tensor(torch.empty_strided(shape, strides, dtype=dtype))
        allocations.append(value)
        return value
    def reinterpret(tensor, shape, strides, offset):
        calls.append("reinterpret")
        return Tensor(torch.as_strided(tensor.data, shape, strides, tensor.storage_offset() + offset))
    def stream(device):
        assert current[0] == device == 0
        calls.append("stream")
        return 0
    namespace.update(torch=SimpleNamespace(Tensor=Tensor, strided=torch.strided,
        float32=torch.float32, bfloat16=torch.bfloat16,
        cuda=SimpleNamespace(_DeviceGuard=DeviceGuard, set_device=set_device)),
        triton=SimpleNamespace(cdiv=lambda n, b: (n+b-1)//b),
        empty_strided_cuda=allocate, reinterpret_tensor=reinterpret, get_raw_stream=stream)
    namespace[KERNEL] = Launch()
    nodes = [function(SOURCE, "_check_inputs"), function(SOURCE, WRAPPER)]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace[WRAPPER], calls, allocations, current, selected, compile_kernel, pid


def unaligned(shape):
    size = 4
    for d in shape: size *= d
    storage = torch.empty(size + 16, dtype=torch.uint8).untyped_storage()[4:4+size]
    return torch.empty(0, dtype=torch.float32).set_(storage, 0, shape)


def checks():
    wrapper, calls, allocations, current, selected, compile_kernel, pid = interpreter()
    w, bias, x = Tensor(torch.ones(3)), Tensor(torch.zeros(3)), Tensor(torch.ones(2, 3))
    rejected = [(None, bias, x), (w, None, x), (w, bias, None),
        (Tensor(w.data.half()), bias, x), (w, bias, Tensor(x.data.bfloat16())),
        (Tensor(w.data, cuda=False), bias, x), (w, bias, Tensor(x.data, device=1)),
        (w, bias, Tensor(torch.ones(3))), (w, bias, Tensor(torch.empty(2, 0))),
        (Tensor(torch.ones(2)), bias, x), (w, Tensor(torch.ones(1, 3)), x),
        (w, bias, Tensor(torch.ones(3, 2).T)), (Tensor(torch.ones(6)[::2]), bias, x),
        (Tensor(torch.ones(4)[1:]), bias, x), (w, bias, Tensor(torch.ones(9)[3:].reshape(2, 3))),
        (Tensor(torch._neg_view(w.data)), bias, x), (w, bias, Tensor(torch._neg_view(x.data))),
        (w, bias, Tensor(unaligned((2, 3)))), (Tensor(unaligned((3,))), bias, x),
        (w, bias, Tensor(x.data, shape=(2**30, 3), strides=(3, 1)))]
    huge = 2**31 - 2048
    rejected.append((Tensor(w.data, shape=(huge,), strides=(1,)),
        Tensor(bias.data, shape=(huge,), strides=(1,)),
        Tensor(x.data, shape=(0, huge), strides=(huge, 1))))
    for args in rejected:
        calls.clear()
        try: wrapper(*args)
        except (ValueError, TypeError): pass
        else: raise AssertionError("unsupported call accepted")
        assert calls == [], "rejection must precede device guard and allocation"
    admitted = 0
    for block in (1024, 2048):
        selected[0] = block
        for S, D in ((0, 3), (1, 1), (2, 3), (2, 1023), (1, 1024), (2, 1025), (1, 2048), (2, 2049)):
            values = (torch.arange(S*D, dtype=torch.float32).reshape(S, D) % 7) - 3
            weight = torch.linspace(.5, 1.5, D)
            bias_data = torch.linspace(-.25, .25, D)
            args = (Tensor(weight), Tensor(bias_data), Tensor(values))
            before = [raw(t.data).clone() for t in args]
            calls.clear(); allocations.clear()
            y, returned_x, mean, inverse = wrapper(*args)
            mu = values.mean(1, keepdim=True)
            rstd = torch.rsqrt(((values-mu)**2).mean(1, keepdim=True) + 1e-5)
            expected = (((values-mu)*rstd)*weight+bias_data).to(torch.bfloat16)
            torch.testing.assert_close(y.data, expected, rtol=0.01, atol=.01)
            torch.testing.assert_close(mean.data, mu, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(inverse.data, rstd, rtol=1e-5, atol=1e-6)
            assert returned_x is args[2] and y is allocations[2] and mean is allocations[0]
            assert inverse.data.untyped_storage()._cdata == allocations[1].data.untyped_storage()._cdata
            assert inverse.stride() == (1, 1) and mean.stride() == (1, S)
            assert y.dtype == torch.bfloat16 and y.stride() == (D, 1)
            assert inverse.storage_offset() == 0 and current[0] == 1
            assert calls == ["guard", "device", "allocation", "allocation", "reinterpret", "allocation", "stream", "launch"]
            for t, saved in zip(args, before): torch.testing.assert_close(raw(t.data), saved, rtol=0, atol=0)
            admitted += 1
    # Read-only caller storage may alias all three inputs.
    base = Tensor(torch.tensor([[1., 2., 3.]]))
    alias = Tensor(base.data.reshape(3))
    wrapper(alias, alias, base)
    admitted += 1
    # Constant rows expose both original undefined padding and the insufficiency
    # of zero-filling loads without masking the centered square.
    inputs = [Tensor(torch.empty(1, 1)), Tensor(torch.empty(1, 1)),
              Tensor(torch.full((1, 3), 2.)), Tensor(torch.ones(3)),
              Tensor(torch.zeros(3)), Tensor(torch.empty(1, 3, dtype=torch.bfloat16))]
    pid[0] = 0
    original = compile_kernel(function(ORIGINAL, KERNEL))
    original(*(Pointer(t) for t in inputs), 1, 3, 1, 1024)
    assert torch.isnan(inputs[0].data).all()
    zero_only = function(SOURCE, KERNEL)
    square = next(n for n in ast.walk(zero_only) if isinstance(n, ast.Assign)
                  and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "tmp10")
    square.value = ast.parse("tmp9 * tmp9", mode="eval").body
    compile_kernel(zero_only)(*(Pointer(t) for t in inputs), 1, 3, 1, 1024)
    assert inputs[1].data.item() < 1, "unmasked padded centered squares corrupt the variance"
    compile_kernel(function(SOURCE, KERNEL))(*(Pointer(t) for t in inputs), 1, 3, 1, 1024)
    assert inputs[1].data.item() > 300 and torch.equal(inputs[-1].data, torch.zeros_like(inputs[-1].data))
    print(f"Passed: exact three-site kernel repair, preserved host operations, {len(rejected)} rejections, {admitted} admitted CPU source executions, two failing tail controls.")
    print("Both RBLOCK choices and autotune trials were exercised. CUDA was mocked; no GPU numerical conformance is claimed.")


if __name__ == "__main__":
    source_checks()
    checks()

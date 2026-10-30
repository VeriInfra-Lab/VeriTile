"""Actual kernel AST and host checks on CPU; CUDA calls are mocked."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "layer_norm_ops_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/layer_norm_ops/layer_norm_ops.py"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_checks():
    forward = "_layer_norm_fwd_1pass_kernel"
    backward = "_layer_norm_bwd_kernel"
    assert ast.dump(function(SOURCE, forward)) == ast.dump(function(ORIGINAL, forward))
    original = function(ORIGINAL, backward)
    index = next(i for i, arg in enumerate(original.args.args) if arg.arg == "M")
    original.args.args[index:index] = [ast.arg(arg="stride_mean"), ast.arg(arg="stride_rstd")]
    for node in ast.walk(original):
        if not isinstance(node, ast.Call) or ast.unparse(node.func) != "tl.load": continue
        address = ast.unparse(node.args[0])
        if address == "W + cols": node.keywords.append(ast.keyword(arg="other", value=ast.Constant(0.0)))
        if address == "Mean + row": node.args[0] = ast.parse("Mean + row * stride_mean", mode="eval").body
        if address == "Rstd + row": node.args[0] = ast.parse("Rstd + row * stride_rstd", mode="eval").body
    assert ast.dump(original) == ast.dump(function(SOURCE, backward))
    for name in ("_layer_norm_fwd", "_layer_norm_bwd"):
        before, after = function(ORIGINAL, name), function(SOURCE, name)
        after.body.pop(0)
        if name.endswith("bwd"):
            query = next(n for n in before.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "sm_count")
            slot = next(i for i, n in enumerate(after.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_dw")
            after.body.insert(slot, query)
            launch = next(n for n in ast.walk(after) if isinstance(n, ast.Call)
                          and isinstance(n.func, ast.Subscript) and ast.unparse(n.func.value) == backward)
            for _ in range(2): launch.args.pop(18)
        assert ast.dump(before) == ast.dump(after), "required source host operations must remain"
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()
    text = (HERE / "Contract.lean").read_text()
    for required in ["ScalarContract", "0 < q", "rowWidth", "rowsPerProgram", "blockSize", "currentDevice = 0"]:
        assert required in text


class Tensor:
    events = []
    def __init__(self, data, *, device=0, cuda=True, shape=None, strides=None):
        self.data, self.device, self.is_cuda = data, torch.device("cuda", device), cuda
        self.shape_override, self.strides_override = shape, strides
    @property
    def shape(self): return tuple(self.data.shape) if self.shape_override is None else self.shape_override
    @property
    def ndim(self): return len(self.shape)
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    def stride(self, axis=None):
        value = self.data.stride() if self.strides_override is None else self.strides_override
        return value if axis is None else value[axis]
    def storage_offset(self): return self.data.storage_offset()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return math.prod(self.shape)
    def element_size(self): return self.data.element_size()
    def data_ptr(self):
        assert self.numel() > 0
        return self.data.data_ptr()
    def sum(self, axis):
        self.events.append("host sum")
        return Tensor(self.data.sum(axis), device=self.device.index)
    def to(self, dtype):
        self.events.append("host cast")
        return self if self.dtype == dtype else Tensor(self.data.to(dtype), device=self.device.index)


def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))


class Pointer:
    def __init__(self, tensor, offset=0): self.tensor, self.offset = tensor, offset
    def __add__(self, offset): return Pointer(self.tensor, self.offset + offset)


def interpreter():
    current, sm_count, pid, events = [0], [4], [0], []
    Tensor.events = events
    def load(ptr, mask=None, other=float("nan")):
        offsets = torch.as_tensor(ptr.offset)
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        values = torch.full(offsets.shape, other, dtype=ptr.tensor.dtype)
        values[mask] = raw(ptr.tensor.data)[ptr.tensor.storage_offset() + offsets[mask]]
        return values
    def store(ptr, values, mask=None):
        offsets = torch.as_tensor(ptr.offset)
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        raw(ptr.tensor.data)[ptr.tensor.storage_offset()+offsets[mask]] = torch.as_tensor(values).expand(offsets.shape)[mask].to(ptr.tensor.dtype)
    tl = SimpleNamespace(program_id=lambda axis: pid[0], arange=torch.arange, float32=torch.float32,
        load=load, store=store, zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        sum=lambda value, axis: value.sum(axis), sqrt=torch.sqrt, where=torch.where)
    namespace = {"tl": tl, "math": math}
    kernels = {}
    def compile_kernel(node):
        node = copy.deepcopy(node)
        node.decorator_list = []
        for arg in node.args.args: arg.annotation = None
        scope = dict(namespace)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), scope)
        return scope[node.name]
    class Launch:
        def __init__(self, name, pointer_count): self.name, self.pointer_count = name, pointer_count
        def __getitem__(self, grid):
            def run(*args):
                assert current[0] == 0
                events.append(self.name)
                values = [Pointer(value) if isinstance(value, Tensor) else value for value in args[:self.pointer_count]]
                values += list(args[self.pointer_count:])
                if self.pointer_count == 12: values.append(args[3] is not None)
                for program in range(grid[0]):
                    pid[0] = program
                    kernels[self.name](*values)
            return run
    for name, count in (("_layer_norm_fwd_1pass_kernel", 8), ("_layer_norm_bwd_kernel", 12)):
        kernels[name] = compile_kernel(function(SOURCE, name))
        namespace[name] = Launch(name, count)
    class Device:
        def __init__(self, index): self.index = index
        def __enter__(self):
            self.old, current[0] = current[0], self.index
            events.append("device")
        def __exit__(self, *exc): current[0] = self.old
    def empty(*shape, dtype, device):
        events.append("allocation")
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        index = current[0] if device == "cuda" else device.index
        return Tensor(torch.empty(shape, dtype=dtype), device=index)
    def empty_like(x, dtype=None):
        events.append("allocation")
        return Tensor(torch.empty_like(x.data, dtype=dtype), device=x.device.index)
    def properties(device):
        events.append("properties")
        assert device.index == 0
        return SimpleNamespace(multi_processor_count=sm_count[0])
    namespace.update(torch=SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
        empty=empty, empty_like=empty_like,
        cuda=SimpleNamespace(current_device=lambda: current[0], device=Device, get_device_properties=properties)),
        triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length()))
    names = ("_check_tensor", "_check_common", "_check_forward", "_check_backward", "_layer_norm_fwd", "_layer_norm_bwd")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function(SOURCE, name) for name in names], type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace, events, current, sm_count, compile_kernel, pid


def checks():
    ns, events, current, sm_count, compile_kernel, pid = interpreter()
    fwd, bwd = ns["_layer_norm_fwd"], ns["_layer_norm_bwd"]
    x, w, bias = Tensor(torch.ones(2, 3)), Tensor(torch.ones(3)), Tensor(torch.zeros(3))
    rejects = [lambda: fwd(None, w, bias, 1e-6), lambda: fwd(x, None, bias, 1e-6),
        lambda: fwd(Tensor(torch.ones(3)), w, bias, 1e-6),
        lambda: fwd(Tensor(torch.empty(2, 0)), Tensor(torch.empty(0)), None, 1e-6),
        lambda: fwd(Tensor(torch.empty(0, 16385)), Tensor(torch.empty(16385)), None, 1e-6),
        lambda: fwd(Tensor(x.data.half()), w, bias, 1e-6),
        lambda: fwd(x, Tensor(w.data, device=1), bias, 1e-6),
        lambda: fwd(Tensor(x.data, cuda=False), w, bias, 1e-6),
        lambda: fwd(Tensor(torch.ones(3, 2).T), w, bias, 1e-6),
        lambda: fwd(x, Tensor(torch.ones(4)[1:]), bias, 1e-6),
        lambda: fwd(Tensor(torch._neg_view(x.data)), w, bias, 1e-6),
        lambda: fwd(x, w, bias, 1e-6, out_dtype=torch.float16),
        lambda: fwd(x, w, bias, 1e-6, residual_dtype=torch.float16),
        lambda: fwd(x, w, bias, 1e-6, residual=Tensor(torch.ones(1, 3))),
        lambda: fwd(x, w, bias, 1e-6, is_rms_norm=1),
        lambda: fwd(Tensor(x.data, shape=(2**30, 3), strides=(3, 1)), w, bias, 1e-6)]
    rejects += [lambda p=p: fwd(x, w, bias, p) for p in (0.0, -0.0, -1., float("nan"), float("inf"), 1, True)]
    mean, rstd = Tensor(torch.ones(2)), Tensor(torch.ones(2))
    rejects += [lambda: bwd(x, x, w, bias, 1e-6, None, rstd),
        lambda: bwd(x, x, w, bias, 1e-6, mean, rstd, is_rms_norm=True),
        lambda: bwd(x, x, w, bias, 1e-6, mean, Tensor(torch.ones(4)[::2])),
        lambda: bwd(x, x, w, bias, 1e-6, mean, rstd, x_dtype=torch.float16),
        lambda: bwd(x, x, w, bias, 1e-6, mean, rstd, has_residual=1),
        lambda: bwd(x, x, w, bias, 1e-6, mean, rstd, recompute_output=1)]
    for rejected in rejects:
        events.clear()
        try: rejected()
        except (ValueError, TypeError): pass
        else: raise AssertionError("unsupported call accepted")
        assert "allocation" not in events and "device" not in events
    current[0] = 1
    try: fwd(x, w, bias, 1e-6)
    except ValueError: pass
    else: raise AssertionError("current-device mismatch accepted")
    current[0] = 0
    admitted = 0
    for M, N in ((0, 3), (1, 1), (2, 3), (5, 17)):
        for rms, residual, has_bias, recompute in ((False, False, False, False), (False, True, True, True),
                (True, False, True, True), (True, True, False, False)):
            data = (torch.arange(M*N, dtype=torch.float32).reshape(M, N) % 7 - 3) / 4
            weight, bias_data = torch.linspace(.5, 1.5, N), torch.linspace(-.2, .2, N)
            x = Tensor(data); w = Tensor(weight); bias = Tensor(bias_data) if has_bias else None
            res = Tensor(torch.full_like(data, .25)) if residual else None
            events.clear()
            y, mean, rstd, z = fwd(x, w, bias, 1e-3, residual=res,
                out_dtype=torch.float32, residual_dtype=torch.float64 if residual else None, is_rms_norm=rms)
            zref = data + .25 if residual else data
            mu = torch.zeros((M, 1)) if rms else zref.mean(1, keepdim=True)
            inv = torch.rsqrt(((zref-mu)**2).mean(1, keepdim=True) + 1e-3)
            xhat = (zref-mu)*inv
            yref = xhat*weight + (bias_data if has_bias else 0)
            torch.testing.assert_close(y.data, yref, rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(rstd.data, inv.flatten(), rtol=2e-5, atol=2e-5)
            assert (mean is None) == rms
            if mean is not None: torch.testing.assert_close(mean.data, mu.flatten())
            assert (z is x) == (not residual)
            dy = Tensor(torch.full_like(data, .3))
            dres = Tensor(torch.full_like(data, -.1)) if residual else None
            result = bwd(dy, z, w, bias, 1e-3, mean, rstd, dresidual=dres,
                has_residual=residual, is_rms_norm=rms, x_dtype=torch.float32, recompute_output=recompute)
            dx, dw, db, dres_in = result[:4]
            wdy = weight * dy.data
            c1 = (xhat*wdy).mean(1, keepdim=True)
            c2 = 0 if rms else wdy.mean(1, keepdim=True)
            dxref = (wdy - (xhat*c1+c2))*inv + (-.1 if residual else 0)
            torch.testing.assert_close(dx.data, dxref, rtol=2e-4, atol=2e-5)
            torch.testing.assert_close(dw.data, (dy.data*xhat).sum(0), rtol=2e-5, atol=2e-5)
            if has_bias: torch.testing.assert_close(db.data, dy.data.sum(0))
            else: assert db is None
            assert dres_in is (dx if residual else None)
            if recompute: torch.testing.assert_close(result[4].data, y.data, rtol=0, atol=0)
            assert events.count("properties") == 1
            assert events.count("host sum") == events.count("host cast") == (2 if has_bias else 1)
            torch.testing.assert_close(x.data, data, rtol=0, atol=0)
            assert current[0] == 0
            admitted += 1
    # Original backward undefined W padding contaminates the row reductions.
    M, N, B = 1, 3, 4
    inputs = [Tensor(torch.tensor([[1., 2., 3.]])), Tensor(torch.ones(N)), None, None,
        Tensor(torch.ones(M, N)), Tensor(torch.empty(M, N)), Tensor(torch.empty(1, N)),
        None, None, None, Tensor(torch.tensor([2.])), Tensor(torch.tensor([1.]))]
    values = [Pointer(t) if isinstance(t, Tensor) else t for t in inputs]
    pid[0] = 0
    compile_kernel(function(ORIGINAL, "_layer_norm_bwd_kernel"))(*values,
        N, 0, N, N, 0, 0, M, N, 1e-3, 1, False, B, False, False, False, False)
    assert torch.isnan(inputs[5].data).all()
    compile_kernel(function(SOURCE, "_layer_norm_bwd_kernel"))(*values,
        N, 0, N, N, 0, 0, 1, 1, M, N, 1e-3, 1, False, B, False, False, False, False)
    assert torch.isfinite(inputs[5].data).all()
    print(f"Passed: exact source/host repair checks, {len(rejects)+1} rejected calls, {admitted} admitted composed CPU source executions, original-tail failure control.")
    print("Source host sums/casts, optional branches and alias returns were exercised. CUDA is mocked; no native GPU claim.")


if __name__ == "__main__":
    source_checks()
    checks()

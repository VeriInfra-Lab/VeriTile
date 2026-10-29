"""Exact source changes and admitted CPU source executions; CUDA is mocked."""
import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "layer_norm_fwd_repaired.py"
ORIGINAL = HERE.parents[2] / "tritonbench_g/layer_norm_fwd/layer_norm_fwd.py"
KERNEL = "_layer_norm_fwd_1pass_kernel"


def function(path, name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_checks():
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest in (HERE / "Proofs/CompilerProfile.lean").read_text(), "compiler provenance must use this source"
    original, repaired = function(ORIGINAL, KERNEL), function(SOURCE, KERNEL)
    assert ast.unparse(repaired.body[0]) == "row = tl.program_id(0).to(tl.int64)"
    repaired.body[0].value = repaired.body[0].value.func.value
    assert ast.dump(original) == ast.dump(repaired)
    original, repaired = function(ORIGINAL, "_layer_norm_fwd"), function(SOURCE, "_layer_norm_fwd")
    assert ast.unparse(repaired.body.pop(0).value.func) == "_check_inputs"
    assert ast.dump(original) == ast.dump(repaired), "all original host operations and branches must remain"
    assert ".Proofs." not in (HERE / "Kernels.lean").read_text()


def uniform(seed, col):
    mask = (1 << 32) - 1
    c0, c1, c2, c3 = col, 0, 0, 0
    k0, k1 = int(seed) & mask, 0
    for _ in range(7):
        a, b = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((b >> 32) ^ c1 ^ k0) & mask, b & mask, ((a >> 32) ^ c3 ^ k1) & mask, a & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    signed = c0 if c0 < 1 << 31 else c0 - (1 << 32)
    mag = -signed - 1 if signed < 0 else signed
    return float(np.float32(np.float32(mag) * np.float32(4.6566127342e-10)))


class Tensor:
    def __init__(self, data, *, device=0, cuda=True, strides=None):
        self.data, self.device = data, torch.device("cuda" if cuda else "cpu", device if cuda else None)
        self.strides_override = strides
    @property
    def shape(self): return tuple(self.data.shape)
    @property
    def ndim(self): return self.data.ndim
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    def stride(self, axis=None):
        value = self.data.stride() if self.strides_override is None else self.strides_override
        return value if axis is None else value[axis]
    def storage_offset(self): return self.data.storage_offset()
    def is_neg(self): return self.data.is_neg()
    def is_contiguous(self): return self.data.is_contiguous()
    def numel(self): return self.data.numel()
    def element_size(self): return self.data.element_size()
    def data_ptr(self):
        assert self.numel() > 0
        return self.data.data_ptr()
    def tensor_split(self, count, dim):
        return tuple(Tensor(t, device=self.device.index) for t in self.data.tensor_split(count, dim))


def raw(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))


class Pointer:
    def __init__(self, tensor, offset=0): self.tensor, self.offset = tensor, offset
    def __add__(self, offset):
        left, right = self.offset, offset
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            dtype = torch.promote_types(left.dtype, right.dtype)
            left, right = left.to(dtype), right.to(dtype)
        return Pointer(self.tensor, left + right)


class CompilationFailure(RuntimeError): pass
class StopAtLoad(Exception): pass


def interpreter():
    current, pid, events = [0], [0], []
    stop = [False]
    address = []
    def load(ptr, mask=None, other=float("nan")):
        if stop[0]:
            address.append(torch.as_tensor(ptr.offset).clone())
            raise StopAtLoad()
        offsets = torch.as_tensor(ptr.offset).long()
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        # Unspecified inactive values are a permitted source behavior; inactive
        # affine lanes never contribute to reductions or active stores.
        fill = other if ptr.tensor.dtype.is_floating_point else 0
        values = torch.full(offsets.shape, fill, dtype=ptr.tensor.dtype)
        values[mask] = raw(ptr.tensor.data)[ptr.tensor.storage_offset()+offsets[mask]]
        return values
    def store(ptr, values, mask=None):
        offsets = torch.as_tensor(ptr.offset).long()
        if mask is None: mask = torch.ones_like(offsets, dtype=torch.bool)
        raw(ptr.tensor.data)[ptr.tensor.storage_offset()+offsets[mask]] = torch.as_tensor(values).expand(offsets.shape)[mask].to(ptr.tensor.dtype)
    def rand(seed, cols, n_rounds):
        assert n_rounds == 7
        return torch.tensor([uniform(int(seed), int(c)) for c in cols], dtype=torch.float32)
    tl = SimpleNamespace(program_id=lambda axis: torch.tensor(pid[0], dtype=torch.int32),
        arange=lambda start, end: torch.arange(start, end, dtype=torch.int32),
        float32=torch.float32, int64=torch.int64, uint32=torch.uint32,
        load=load, store=store, sum=lambda value, axis: value.sum(axis), sqrt=torch.sqrt,
        where=torch.where, rand=rand)
    namespace = {"tl": tl, "math": math}
    def compile_kernel(node):
        node = copy.deepcopy(node)
        node.decorator_list = []
        for arg in node.args.args: arg.annotation = None
        scope = dict(namespace)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(SOURCE), "exec"), scope)
        return scope[node.name]
    kernel = compile_kernel(function(SOURCE, KERNEL))
    class Launch:
        def __getitem__(self, grid):
            def run(*args):
                events.append("compile")
                assert current[0] == args[0].device.index
                assert args[28] == (args[9] is not None)
                if args[26] == 0:
                    raise CompilationFailure("arange requires positive power-of-two end")
                values = [Pointer(value) if isinstance(value, Tensor) else value for value in args[:15]] + list(args[15:])
                values += [args[5] is not None, args[6] is not None, args[7] is not None]
                # Three actual predecessor executions exercise autotune replay.
                for trial in range(3):
                    events.append("launch")
                    for program in range(grid[0]):
                        pid[0] = program
                        kernel(*values)
            return run
    class Device:
        def __init__(self, index): self.index = index
        def __enter__(self):
            self.old, current[0] = current[0], self.index
            events.append("enter")
        def __exit__(self, *exc):
            current[0] = self.old
            events.append("exit")
    def empty(*shape, dtype, device):
        events.append("allocate")
        if len(shape) == 1 and isinstance(shape[0], tuple): shape = shape[0]
        return Tensor(torch.empty(shape, dtype=dtype), device=device.index)
    def empty_like(x, dtype=None):
        events.append("allocate")
        return Tensor(torch.empty_like(x.data, dtype=dtype), device=x.device.index)
    def randint(high, shape, device, dtype):
        events.append("seeds")
        return Tensor(torch.randint(high, shape, dtype=dtype), device=device.index)
    namespace.update(torch=SimpleNamespace(Tensor=Tensor, dtype=torch.dtype, strided=torch.strided,
        float32=torch.float32, int64=torch.int64, bool=torch.bool,
        empty=empty, empty_like=empty_like, randint=randint, cuda=SimpleNamespace(device=Device)),
        triton=SimpleNamespace(next_power_of_2=lambda n: 0 if n == 0 else 1 << (n-1).bit_length()))
    namespace[KERNEL] = Launch()
    nodes = [function(SOURCE, name) for name in ("_check_tensor", "_check_inputs", "_layer_norm_fwd")]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), "exec"), namespace)
    return namespace, events, current, compile_kernel, pid, stop, address


def checks():
    ns, events, current, compile_kernel, pid, stop, address = interpreter()
    run = ns["_layer_norm_fwd"]
    x, w = Tensor(torch.ones(2, 3)), Tensor(torch.ones(3))
    rejects = [lambda: run(None, w, None, 1.0), lambda: run(x, None, None, 1.0),
        lambda: run(Tensor(torch.ones(3)), w, None, 1.0),
        lambda: run(Tensor(torch.ones(2, 3).half()), w, None, 1.0),
        lambda: run(Tensor(torch.ones(2, 3), cuda=False), w, None, 1.0),
        lambda: run(x, Tensor(torch.ones(3), device=1), None, 1.0),
        lambda: run(x, Tensor(torch.ones(4)[1:]), None, 1.0),
        lambda: run(Tensor(torch.ones(3, 2).T), w, None, 1.0),
        lambda: run(Tensor(torch._neg_view(torch.ones(2, 3))), w, None, 1.0),
        lambda: run(x, w, None, 1.0, x1=x, rowscale=Tensor(torch.ones(2))),
        lambda: run(x, w, None, 1.0, residual=object()),
        lambda: run(x, w, Tensor(torch.ones(4)), 1.0),
        lambda: run(x, w, None, 1.0, is_rms_norm=1),
        lambda: run(x, w, None, 1.0, return_dropout_mask=1),
        lambda: run(x, w, None, 1.0, out_dtype="fp32"),
        lambda: run(x, w, None, 1.0, residual_dtype="fp32")]
    rejects += [lambda e=e: run(x, w, None, e) for e in [0.0, -0.0, -1.0, float("nan"), float("inf"), 1, True]]
    rejects += [lambda p=p: run(x, w, None, 1.0, dropout_p=p) for p in [-0.1, 1.0, float("nan"), float("inf"), 0, False]]
    for reject in rejects:
        events.clear()
        try: reject()
        except (ValueError, TypeError): pass
        else: raise AssertionError("bad call admitted")
        assert not events, "rejection must precede allocation or device effects"
    cases = []
    for rms in [False, True]:
        for mode in ["basic", "dropout", "two", "scale", "residual", "dual"]:
            cases.append((3, 5, rms, mode, torch.float32, None))
    cases += [(0, 3, False, "two", torch.float32, None), (2, 1, True, "basic", torch.float32, None),
        (2, 128, False, "dual", torch.float16, torch.bfloat16),
        (2, 7, False, "scale", torch.bfloat16, torch.float64),
        (2, 3, True, "residual", torch.float64, "ignored override"),
        (2, 3, False, "alias", torch.float32, None),
        (2, 3, False, "basic", torch.int32, None), (2, 3, False, "basic", torch.bool, None)]
    for M, N, rms, mode, out_dtype, residual_dtype in cases:
        device = 2
        x = Tensor(torch.linspace(-2, 3, max(M*N, 1))[:M*N].reshape(M, N), device=device)
        w = Tensor(torch.linspace(.5, 1.5, N), device=device)
        bias = None if mode == "basic" else Tensor(torch.linspace(-.3, .3, N), device=device)
        extras = dict(residual=None, x1=None, weight1=None, bias1=None, rowscale=None)
        if mode in ["two", "residual", "alias"]: extras["residual"] = x if mode == "alias" else Tensor(torch.full((M,N), .125), device=device)
        if mode in ["two", "alias"]: extras["x1"] = x if mode == "alias" else Tensor(torch.full((M,N), .25), device=device)
        if mode in ["two", "dual", "alias"]:
            extras["weight1"] = w
            extras["bias1"] = bias
        if mode == "scale": extras["rowscale"] = Tensor(torch.linspace(.5, 1.5, M), device=device)
        p = .25 if mode in ["two", "dropout", "scale"] else -0.0
        snapshot = [(t, t.data.clone()) for t in [x,w,bias,*extras.values()] if t is not None]
        events.clear(); current[0] = 1; torch.manual_seed(123)
        out = run(x, w, bias, 1.0, **extras, is_rms_norm=rms, dropout_p=p,
                  return_dropout_mask=True, out_dtype=out_dtype, residual_dtype=residual_dtype)
        assert current[0] == 1 and events.count("launch") == 3
        y,y1,mean,rstd,resout,seeds,m0,m1 = out
        combined = x.data.double()
        if extras["rowscale"] is not None: combined = combined * extras["rowscale"].data.double()[:,None]
        masks = []
        if p > 0:
            for stream in range(2 if extras["x1"] is not None else 1):
                masks.append(torch.tensor([[uniform(seeds.data[stream*M+i].item(), j) > p for j in range(N)] for i in range(M)],dtype=torch.bool).reshape(M,N))
            combined = torch.where(masks[0], combined/(1-p), 0.)
        if extras["x1"] is not None:
            second = extras["x1"].data.double()
            if p > 0: second = torch.where(masks[1], second/(1-p), 0.)
            combined = combined + second
        if extras["residual"] is not None: combined = combined + extras["residual"].data.double()
        mu = combined.mean(1)
        centered = combined if rms else combined-mu[:,None]
        inv = (centered.square().mean(1)+1).rsqrt()
        hat = centered*inv[:,None]
        expected = hat*w.data.double() + (bias.data.double() if bias is not None else 0.)
        tolerance = 8e-3 if out_dtype == torch.bfloat16 else 2e-3 if out_dtype == torch.float16 else 3e-6
        torch.testing.assert_close(y.data.double(), expected.to(out_dtype).double(), rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(rstd.data.double(), inv, rtol=3e-6, atol=3e-6)
        if rms: assert mean is None
        else: torch.testing.assert_close(mean.data.double(), mu, rtol=3e-6, atol=3e-6)
        if extras["weight1"] is not None:
            torch.testing.assert_close(y1.data.double(), expected.to(out_dtype).double(), rtol=tolerance, atol=tolerance)
        else: assert y1 is None
        if extras["residual"] is not None or extras["x1"] is not None or extras["rowscale"] is not None or p > 0 or residual_dtype is not None:
            dtype = torch.float32 if extras["residual"] is not None else residual_dtype or torch.float32
            assert resout.dtype == dtype
            torch.testing.assert_close(resout.data.double(), combined.to(dtype).double(), rtol=3e-6, atol=3e-6)
        else: assert resout is x
        if p > 0:
            assert torch.equal(m0.data, masks[0])
            if extras["x1"] is not None:
                assert torch.equal(m1.data, masks[1])
                assert m1.storage_offset() == M*N
                assert m0.data.untyped_storage().data_ptr() == m1.data.untyped_storage().data_ptr()
            else: assert m1 is None
        else: assert seeds is None and m0 is None and m1 is None
        for t, saved in snapshot: assert torch.equal(t.data, saved)
    # Source failure paths preserve original allocation-before-width order,
    # zero-column compile failure and device restoration after that failure.
    for M,N,error in [(2,0,CompilationFailure),(0,16385,RuntimeError)]:
        events.clear(); current[0] = 1
        try: run(Tensor(torch.empty(M,N)), Tensor(torch.ones(N)), None, 1.0)
        except error: pass
        else: raise AssertionError("expected original failure branch")
        assert events[0] == "allocate" and current[0] == 1
        if N == 0: assert events[-1] == "exit"
        else: assert "compile" not in events
    # Sparse address probe: no huge allocation, no read occurs. The actual
    # original/repaired ASTs expose signed32 overflow before their first load.
    ptr = Pointer(Tensor(torch.ones(1)))
    args = [ptr]*15 + [4096]*6 + [1048577,4096,1.0,0.0,False,4096] + [False]*9
    pid[0] = 1048576; stop[0] = True
    for path in [ORIGINAL,SOURCE]:
        try: compile_kernel(function(path,KERNEL))(*args)
        except StopAtLoad: pass
    assert int(address[0][0]) == 0
    assert int(address[1][0]) == 2**32
    print(f"Passed: exact source/host checks, {len(rejects)} rejected calls, {len(cases)} admitted CPU source executions, two original failure paths, signed32 overflow control.")
    print("CUDA, allocation and compilation are mocked; no native GPU numerical claim.")


if __name__ == "__main__":
    source_checks()
    checks()

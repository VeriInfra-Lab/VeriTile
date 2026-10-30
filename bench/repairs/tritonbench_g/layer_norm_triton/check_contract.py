"""Actual repaired source over CPU backing storage; CUDA/runtime are mocked."""
import ast
import copy
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from bind_source import generated

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'layer_norm_triton_repaired.py'
ORIGINAL = HERE.parents[2] / 'tritonbench_g/layer_norm_triton/layer_norm_triton.py'


class Tensor:
    def __init__(self, data, device=0, cuda=True, ptr_delta=0):
        self.data = data
        self.device = torch.device('cuda', device) if cuda else torch.device('cpu')
        self.ptr_delta = ptr_delta
    @property
    def shape(self): return tuple(self.data.shape)
    @property
    def ndim(self): return self.data.ndim
    @property
    def dtype(self): return self.data.dtype
    @property
    def layout(self): return self.data.layout
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def numel(self): return self.data.numel()
    def data_ptr(self):
        assert self.numel(), 'empty pointer must not be queried'
        return self.data.data_ptr() + self.ptr_delta
    def element_size(self): return self.data.element_size()
    def stride(self, axis): return self.data.stride(axis)
    def contiguous(self):
        assert self.data.is_contiguous(), 'guard must reject layout conversion'
        return self
    def reshape(self, *shape): return Tensor(self.data.reshape(*shape), self.device.index)
    def view(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Tensor(self.data.view(*shape), self.device.index)


class Pointer:
    def __init__(self, tensor, offset=0): self.tensor, self.offset = tensor, offset
    @property
    def dtype(self): return SimpleNamespace(element_ty=self.tensor.dtype)
    def __add__(self, offset): return Pointer(self.tensor, self.offset + offset)


def backing(t):
    return torch.empty(0, dtype=t.dtype).set_(t.untyped_storage(), 0,
        (t.untyped_storage().nbytes() // t.element_size(),), (1,))


class Context:
    def save_for_backward(self, *tensors):
        self.saved = tensors
        self.versions = [t.data._version for t in tensors]
    @property
    def saved_tensors(self):
        if self.versions != [t.data._version for t in self.saved]:
            raise RuntimeError('saved tensor version changed')
        return self.saved


def environment(path=SOURCE):
    events, current, pid = [], [1], [0]
    class Device:
        def __init__(self, owner): self.owner = owner.index
        def __enter__(self):
            self.old = current[0]
            current[0] = self.owner
            events.append(('enter', self.owner))
        def __exit__(self, *exc):
            current[0] = self.old
            events.append(('leave', self.old))
    def allocation(kind, shape, dtype, device):
        events.append((kind, tuple(shape), dtype, device.index))
        data = torch.zeros(shape, dtype=dtype) if kind == 'zeros' else torch.empty(shape, dtype=dtype)
        return Tensor(data, device.index)
    def load(p, mask=None, other=None):
        offset = torch.as_tensor(p.offset, dtype=torch.int64)
        mask = torch.ones_like(offset, dtype=torch.bool) if mask is None else torch.as_tensor(mask, dtype=torch.bool)
        offset, mask = torch.broadcast_tensors(offset, mask)
        result = torch.full(offset.shape, float('nan') if other is None else other, dtype=p.tensor.dtype)
        active = offset[mask]
        data = backing(p.tensor.data)
        active = active + p.tensor.data.storage_offset()
        assert bool(((active >= 0) & (active < data.numel())).all())
        result[mask] = data[active]
        return result
    def store(p, value, mask=None):
        offset = torch.as_tensor(p.offset, dtype=torch.int64)
        mask = torch.ones_like(offset, dtype=torch.bool) if mask is None else torch.as_tensor(mask, dtype=torch.bool)
        offset, mask, value = torch.broadcast_tensors(offset, mask, torch.as_tensor(value))
        active = offset[mask] + p.tensor.data.storage_offset()
        data = backing(p.tensor.data)
        assert bool(((active >= 0) & (active < data.numel())).all())
        data[active] = value[mask].to(p.tensor.dtype)
    tl = SimpleNamespace(program_id=lambda axis: torch.tensor(pid[0], dtype=torch.int32),
        int64=torch.int64, float32=torch.float32, constexpr=object(),
        cast=lambda v, dtype: torch.as_tensor(v).to(dtype),
        arange=lambda a, b: torch.arange(a, b, dtype=torch.int32),
        zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        load=load, store=store, sum=lambda x, axis: x.sum(dim=axis), sqrt=torch.sqrt,
        where=lambda c, a, b: torch.where(c, torch.as_tensor(a), torch.as_tensor(b)))
    class Kernel:
        def __init__(self, fn): self.fn = fn
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                assert current[0] == next(v.device.index for v in args if isinstance(v, Tensor))
                events.append(('launch', self.fn.__name__, grid[0], dict(kwargs)))
                for p in range(grid[0]):
                    pid[0] = p
                    self.fn(*(Pointer(v) if isinstance(v, Tensor) else v for v in args),
                            **{k: v for k, v in kwargs.items() if k.startswith('BLOCK') or k == 'GROUP_SIZE_M'})
            return launch
    fake_torch = SimpleNamespace(Tensor=Tensor, strided=torch.strided, float32=torch.float32,
        cuda=SimpleNamespace(device=Device),
        empty=lambda shape, dtype, device: allocation('empty', shape, dtype, device),
        empty_like=lambda t: allocation('empty', t.shape, t.dtype, t.device),
        zeros_like=lambda t: allocation('zeros', t.shape, t.dtype, t.device))
    fake_triton = SimpleNamespace(jit=lambda fn: Kernel(fn),
        next_power_of_2=lambda n: 0 if n == 0 else 1 << (n - 1).bit_length(),
        cdiv=lambda a, b: (a + b - 1) // b)
    ns = dict(math=math, torch=fake_torch, triton=fake_triton, tl=tl)
    module = ast.parse(path.read_text())
    for index, node in enumerate(module.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'layer_norm' for t in node.targets):
            module.body = module.body[:index]
            break
    module.body = [n for n in module.body if not isinstance(n, (ast.Import, ast.ImportFrom))
                   and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'layer_norm' for t in n.targets))]
    for n in module.body:
        if isinstance(n, ast.ClassDef): n.bases = []
    class ValueAssignments(ast.NodeTransformer):
        # Triton variables are SSA values; Python Tensor += would mutate aliases.
        def visit_AugAssign(self, node):
            target = copy.deepcopy(node.target)
            target.ctx = ast.Load()
            return ast.copy_location(ast.Assign(targets=[node.target],
                value=ast.BinOp(left=target, op=node.op, right=node.value)), node)
    module = ValueAssignments().visit(module)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), ns)
    return ns, events, current


def reference(x, w, b, dy, normalized, eps):
    ns = (normalized,) if type(normalized) is int else tuple(normalized)
    m, n = math.prod(x.shape[:-len(ns)]), math.prod(ns)
    if not m or not n:
        return torch.empty_like(x), torch.empty_like(x), torch.zeros_like(w), torch.zeros_like(b)
    z = x.double().reshape(m, n)
    mu = z.mean(1, keepdim=True)
    inv = torch.rsqrt(((z - mu) ** 2).mean(1, keepdim=True) + eps)
    h = (z - mu) * inv
    g = dy.double().reshape(m, n)
    wg = g * w.double().reshape(n)
    y = (h * w.double().reshape(n) + b.double().reshape(n)).reshape(x.shape)
    dx = ((wg - h * (wg * h).mean(1, keepdim=True) - wg.mean(1, keepdim=True)) * inv).reshape(x.shape)
    return y.float(), dx.float(), (g * h).sum(0).float().reshape(w.shape), g.sum(0).float().reshape(b.shape)


def main():
    assert (HERE / 'Proofs/PythonSource.lean').read_text() == generated()
    kernel_pins = {'_layer_norm_fwd_fused': 'e9c81ac37c9826d217a91bf0e3cfa7b2b156f2167a0dff8e521ccd0eff18ade7', '_layer_norm_bwd_dx_fused': '8a951cc64155cc5269a0e9dcfc574e725f56e44da718b4b187fbf62b40371849', '_layer_norm_bwd_dwdb': '54b4f4667d6105cb3e194f20b73a67b60ad9bd367f58c3f512dcef184aea76a6'}
    for node in ast.parse(SOURCE.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name in kernel_pins:
            assert hashlib.sha256(ast.dump(node).encode()).hexdigest() == kernel_pins[node.name]
    ns, events, current = environment()
    layer = ns['LayerNorm']
    torch.set_num_threads(1)
    torch.manual_seed(17)
    x, w, b = Tensor(torch.randn(2, 3)), Tensor(torch.randn(3)), Tensor(torch.randn(3))
    rejected = 0
    def reject(args, cls=Exception):
        nonlocal rejected
        events.clear()
        try: layer.forward(Context(), *args)
        except cls: pass
        else: raise AssertionError('bad inputs admitted')
        assert events == [], events
        rejected += 1
    for bad in [object(), Tensor(torch.randn(2, 3).half()), Tensor(torch.randn(2, 3), cuda=False),
                Tensor(torch.randn(3, 2).t()), Tensor(torch._neg_view(torch.randn(2, 3))),
                Tensor(torch.randn(2, 3), ptr_delta=1)]:
        reject((bad, (3,), w, b, 1e-5))
    for arg in [(), (2,), (2, 3, 4), -1, True, '3', (3.0,), (True,), (-1,), None]:
        reject((x, arg, w, b, 1e-5))
    for eps in [0.0, -0.0, -1.0, float('nan'), float('inf'), 1, True]:
        reject((x, (3,), w, b, eps))
    reject((x, (3,), Tensor(w.data, device=2), b, 1e-5))
    reject((x, (3,), w, Tensor(torch.zeros(2)), 1e-5))
    reject((x, (3,), w, object(), 1e-5))
    admitted = 0
    shapes = [((3,), 3), ((2, 3), (3,)), ((2, 3, 4), (3, 4)), ((2, 3, 4), [4]),
              ((0, 5), (5,)), ((2, 0), (0,)), ((0,), 0), ((2, 0, 3), (0, 3)),
              ((7, 1), 1), ((513, 3), 3), ((2, 17), 17), ((1, 1025), 1025),
              ((1, 4097), 4097), ((1, 8193), 8193), ((1, 16384), 16384)]
    for shape, normalized in shapes:
        affine = (normalized,) if type(normalized) is int else tuple(normalized)
        xd, wd, bd, dyd = torch.randn(shape), torch.randn(affine), torch.randn(affine), torch.randn(shape)
        a, ww, bb, grad = [Tensor(t, device=2) for t in [xd, wd, bd, dyd]]
        expected = reference(xd, wd, bd, dyd, normalized, 1e-5)
        events.clear()
        ctx = Context()
        y = layer.forward(ctx, a, normalized, ww, bb, 1e-5)
        assert current == [1]
        assert ctx.saved[0].data.untyped_storage().data_ptr() == a.data.untyped_storage().data_ptr()
        out = layer.backward(ctx, grad)
        assert current == [1] and out[1] is None and out[4] is None
        for actual, wanted in zip((y.data, out[0].data, out[2].data, out[3].data), expected):
            torch.testing.assert_close(actual, wanted, atol=4e-4, rtol=4e-4, equal_nan=True)
        assert y.shape == shape and out[2].shape == affine and out[3].shape == affine
        if not math.prod(shape):
            assert not any(e[0] == 'launch' for e in events)
        admitted += 1
    # Nonzero storage offsets, live read aliases, and identity contiguous calls.
    base = torch.randn(16)
    a, ww, bb = Tensor(base[2:8].view(2, 3)), Tensor(base[2:5]), Tensor(base[2:5])
    ctx = Context(); y = layer.forward(ctx, a, 3, ww, bb, 1e-5)
    expected = reference(a.data.clone(), ww.data.clone(), bb.data.clone(), a.data.clone(), 3, 1e-5)
    out = layer.backward(ctx, a)
    for actual, wanted in zip((y.data, out[0].data, out[2].data, out[3].data), expected):
        torch.testing.assert_close(actual, wanted, atol=4e-4, rtol=4e-4)
    admitted += 1
    for bad in [Tensor(torch.randn(3, 2)), Tensor(torch.randn(2, 3), device=2),
                Tensor(torch.randn(3, 2).t()), Tensor(torch.randn(2, 3).half())]:
        events.clear()
        try: layer.backward(ctx, bad)
        except (ValueError, TypeError): rejected += 1
        else: raise AssertionError('bad gradient admitted')
        assert not events
    # Saved version failure precedes gradient validation and allocation.
    a.data.add_(1)
    events.clear()
    try: layer.backward(ctx, object())
    except RuntimeError: pass
    else: raise AssertionError('saved version check omitted')
    assert not events
    # Source width failure remains before the device context/allocations.
    events.clear()
    try: layer.forward(Context(), Tensor(torch.empty(1, 16385)), 16385,
                       Tensor(torch.empty(16385)), Tensor(torch.empty(16385)), 1e-5)
    except RuntimeError: pass
    else: raise AssertionError('source width cap changed')
    assert not events
    # Execute the original source: normalized_shape was ignored by its host.
    xd, wd, bd = torch.randn(2, 3, 4), torch.randn(3, 4), torch.randn(3, 4)
    old_ns, _, _ = environment(ORIGINAL)
    old = old_ns['LayerNorm'].forward(Context(), Tensor(xd, device=1), (3, 4),
        Tensor(wd, device=1), Tensor(bd, device=1), 1e-5).data
    target = torch.nn.functional.layer_norm(xd, (3, 4), wd, bd)
    assert not torch.allclose(old, target)
    # The bit domain is not silently narrowed to finite tensor contents.
    special = torch.tensor([[float('nan'), 1.0, -0.0], [float('inf'), -float('inf'), 2.0]])
    a, ww, bb, grad = Tensor(special), Tensor(torch.ones(3)), Tensor(torch.zeros(3)), Tensor(torch.ones(2, 3))
    ctx = Context(); y = layer.forward(ctx, a, 3, ww, bb, 1e-5)
    out = layer.backward(ctx, grad)
    expected = reference(a.data, ww.data, bb.data, grad.data, 3, 1e-5)
    for actual, wanted in zip((y.data, out[0].data, out[2].data, out[3].data), expected):
        torch.testing.assert_close(actual, wanted, atol=4e-4, rtol=4e-4, equal_nan=True)
    admitted += 1
    # Native-size control does not allocate or dereference a huge tensor.
    assert int(np.multiply(np.int32(1 << 20), np.int32(4096), dtype=np.int32)) == 0
    assert int(np.int64(1 << 20) * np.int64(4096)) == 1 << 32
    print(f'Passed: current source AST/hash, {rejected} rejections, {admitted} admitted CPU forward/backward executions, version/width paths, original-axis and signed32 controls.')
    print('CUDA, allocation and compilation are mocked; no native GPU numerical claim.')


if __name__ == '__main__':
    main()

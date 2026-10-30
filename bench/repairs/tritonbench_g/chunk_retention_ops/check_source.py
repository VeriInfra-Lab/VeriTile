"""Execute the actual Python API and JIT source in Triton's CPU interpreter."""
import os
os.environ["TRITON_INTERPRET"] = "1"
import contextlib
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import triton

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "chunk_retention_ops_repaired.py"


class Tensor:
    allocated = []
    def __init__(self, data, device="cuda:1"):
        self.data, self.device = data, torch.device(device)
    shape = property(lambda self: self.data.shape)
    dtype = property(lambda self: self.data.dtype)
    layout = property(lambda self: self.data.layout)
    is_cuda = property(lambda self: self.device.type == "cuda")
    requires_grad = property(lambda self: self.data.requires_grad)
    def dim(self): return self.data.dim()
    def size(self, dim): return self.data.size(dim)
    def stride(self, dim=None): return self.data.stride() if dim is None else self.data.stride(dim)
    def storage_offset(self): return self.data.storage_offset()
    def untyped_storage(self): return self.data.untyped_storage()
    def data_ptr(self): return self.data.data_ptr()
    def is_contiguous(self): return self.data.is_contiguous()
    def is_neg(self): return self.data.is_neg()
    def is_conj(self): return self.data.is_conj()
    def new_empty(self, *shape, dtype=None): return allocate(shape, dtype or self.dtype, self.device)
    def to(self, dtype):
        assert dtype == self.dtype
        return self
    def contiguous(self):
        if self.is_contiguous(): return self
        result = allocate(self.shape, self.dtype, self.device)
        result.data.copy_(self.data)
        return result
    def sum(self, dim):
        value = self.data.sum(dim)
        result = allocate(value.shape, self.dtype, self.device)
        result.data.copy_(value)
        return result


def allocate(shape, dtype, device):
    backing = torch.full((math.prod(shape) + 16,), -444.5, dtype=dtype)
    result = Tensor(backing[8:-8].reshape(shape), device)
    result.data.fill_(float("nan"))
    Tensor.allocated.append((result, backing))
    return result


def bytes_of(x):
    return torch.empty(0, dtype=torch.uint8).set_(x.data.untyped_storage(), 0,
        (x.data.untyped_storage().nbytes(),), (1,)).clone()


def check():
    spec = importlib.util.spec_from_file_location("retention_ops_repaired", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current = [torch.device("cuda:0")]
    launches = []
    @contextlib.contextmanager
    def device_context(device):
        before = current[0]
        current[0] = device
        try: yield
        finally: current[0] = before
    names = ["chunk_retention_fwd_kernel_h", "chunk_retention_fwd_kernel_o",
             "chunk_retention_bwd_kernel_dh", "chunk_retention_bwd_kernel_dqkv"]
    for name in names:
        kernel = getattr(module, name).fn
        class Launch:
            def __init__(self, name, kernel): self.name, self.kernel = name, kernel
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    tensors = [x for x in args if isinstance(x, Tensor)]
                    assert current[0] == tensors[0].device
                    launches.append((self.name, grid))
                    self.kernel[grid](*[x.data if isinstance(x, Tensor) else x for x in args], **kwargs)
                return run
        setattr(module, name, Launch(name, kernel))
    def empty_like(x): return allocate(x.shape, x.dtype, x.device)
    def zeros_like(x):
        result = empty_like(x)
        result.data.zero_()
        return result
    module.torch = SimpleNamespace(Tensor=Tensor, float32=torch.float32, strided=torch.strided,
        empty_like=empty_like, zeros_like=zeros_like, is_grad_enabled=lambda: False,
        cuda=SimpleNamespace(device=device_context))
    original = module.ChunkRetentionFunction
    contexts = []
    class Context:
        def save_for_backward(self, *tensors): self.saved_tensors = tensors
    class Apply:
        @staticmethod
        def apply(*args):
            ctx = Context()
            ctx.needs_input_grad = [isinstance(x, Tensor) and x.requires_grad for x in args]
            contexts.append(ctx)
            return original.forward(ctx, *args)
    module.ChunkRetentionFunction = Apply
    cases = [(1, 1, 1, 1, 1, False, False, 0, "dense"),
             (1, 2, 17, 17, 19, True, True, 0, "dense"),
             (1, 2, 65, 65, 33, True, True, 1, "expanded"),
             (1, 1, 67, 33, 65, True, True, 1, "final_only"),
             (2, 2, 3, 16, 16, False, True, 0, "alias"),
             (1, 1, 64, 32, 32, True, False, 1, "strided"),
             (1, 1, 3, 2, 2, True, True, 0, "no_initial_grad")]
    for B, H, T, K, V, seed, final, checkpoint, mode in cases:
        torch.manual_seed(T + K + V)
        Tensor.allocated.clear()
        def input_tensor(shape):
            backing = torch.randn(math.prod(shape) + 10) * 0.1
            return Tensor(backing[5:-5].reshape(shape).requires_grad_())
        q = input_tensor((B, H, T, K))
        k = q if mode == "alias" else input_tensor(q.shape)
        v = q if mode == "alias" else input_tensor((B, H, T, V))
        h0 = input_tensor((B, H, K, V)) if seed else None
        if mode == "no_initial_grad": h0.data.requires_grad_(False)
        snapshots = [(x, bytes_of(x)) for x in [q, k, v] + ([h0] if seed else [])]
        o, ht = module.chunk_retention(q, k, v, h0, final, None, checkpoint)
        ctx = contexts[-1]
        assert (ctx.saved_tensors[3] is None) == bool(checkpoint)
        if mode == "expanded":
            do = Tensor(torch.tensor(0.2).expand(o.shape))
            dht = Tensor(torch.tensor(0.3).expand(ht.shape))
        elif mode == "strided":
            do = Tensor((torch.randn(B, H, T, V * 2) * 0.1)[..., ::2])
            dht = None
        else:
            do = None if mode == "final_only" else Tensor(torch.randn(o.shape) * 0.1)
            dht = Tensor(torch.randn(ht.shape) * 0.1) if final else None
        snapshots += [(x, bytes_of(x)) for x in [do, dht] if x is not None]
        actual = original.backward(ctx, do, dht)
        assert len(actual) == 7 and actual[4:] == (None, None, None)
        assert (actual[3] is not None) == (seed and mode != "no_initial_grad")
        qr, kr, vr = [x.data.detach().double().requires_grad_() for x in [q, k, v]]
        initial = h0.data.detach().double().requires_grad_() if seed else None
        state = torch.zeros(B, H, K, V, dtype=torch.float64) if initial is None else initial
        decay = (1 - 2. ** (-5 - torch.arange(H, dtype=torch.float64)))[None, :, None, None]
        outputs = []
        for t in range(T):
            state = decay * state + kr[:, :, t, :, None] * vr[:, :, t, None, :]
            outputs.append(K ** -0.5 * torch.einsum("bhkv,bhk->bhv", state, qr[:, :, t]))
        expected = torch.stack(outputs, dim=2)
        loss = (expected * (0 if do is None else do.data.double())).sum()
        if final: loss = loss + (state * dht.data.double()).sum()
        loss.backward()
        torch.testing.assert_close(o.data.double(), expected, rtol=4e-4, atol=2e-6)
        if final: torch.testing.assert_close(ht.data.double(), state, rtol=4e-4, atol=2e-6)
        for got, target in zip(actual[:4], [qr.grad, kr.grad, vr.grad, None if initial is None else initial.grad]):
            if got is not None:
                torch.testing.assert_close(got.data.double(), target, rtol=5e-4, atol=3e-6)
        again = original.backward(ctx, do, dht)
        for a, b in zip(actual[:4], again[:4]):
            if a is not None: assert torch.equal(a.data, b.data)
        for x, before in snapshots: assert torch.equal(bytes_of(x), before)
        for _, backing in Tensor.allocated:
            assert bool((backing[:8] == -444.5).all()) and bool((backing[-8:] == -444.5).all())
        assert current[0] == torch.device("cuda:0")
    rejected = 0
    def make():
        return [Tensor(torch.ones(1, 1, 2, 4)), Tensor(torch.ones(1, 1, 2, 4)),
                Tensor(torch.ones(1, 1, 2, 8)), None, False, None, 1]
    def reject(edit):
        nonlocal rejected
        args = make()
        edit(args)
        count = len(launches)
        try: module.chunk_retention(*args)
        except (TypeError, ValueError): rejected += 1
        else: raise AssertionError("expected rejection")
        assert len(launches) == count
    reject(lambda a: a.__setitem__(0, object()))
    reject(lambda a: a.__setitem__(0, Tensor(torch.ones(1, 1, 2, 4, dtype=torch.float16))))
    reject(lambda a: a.__setitem__(0, Tensor(torch.ones(1, 1, 2, 8)[..., ::2])))
    reject(lambda a: a.__setitem__(0, Tensor(torch.ones(1, 1, 2, 4)._neg_view())))
    reject(lambda a: a.__setitem__(0, Tensor(torch.ones(1, 1, 2, 4), "cuda:0")))
    reject(lambda a: a.__setitem__(1, Tensor(torch.ones(1, 1, 0, 4))))
    reject(lambda a: a.__setitem__(3, Tensor(torch.ones(1, 1, 4, 7))))
    reject(lambda a: a.__setitem__(4, 1))
    reject(lambda a: a.__setitem__(5, float("inf")))
    reject(lambda a: a.__setitem__(5, 1))
    reject(lambda a: a.__setitem__(6, True))
    reject(lambda a: setattr(a[0], "data_ptr", lambda: 3))
    reject(lambda a: a.__setitem__(0, Tensor(torch.as_strided(torch.ones(8), (1, 1, 2, 4), (8, 100, 4, 1)))))
    args = make()
    module.chunk_retention(*args)
    ctx = contexts[-1]
    for mode in ["higher", "terminal", "wrong_do"]:
        count = len(launches)
        module.torch.is_grad_enabled = lambda: mode == "higher"
        try:
            original.backward(ctx, Tensor(torch.ones(1)) if mode == "wrong_do" else Tensor(torch.ones(1, 1, 2, 8)),
                Tensor(torch.ones(1, 1, 4, 8)) if mode == "terminal" else None)
        except ValueError: rejected += 1
        else: raise AssertionError("expected backward rejection")
        assert len(launches) == count
    module.torch.is_grad_enabled = lambda: False
    q, k, v = [Tensor(torch.randn(1, 1, 2, 2) * 0.1) for _ in range(3)]
    do, dht = [Tensor(torch.randn(1, 1, 2, 2) * 0.1) for _ in range(2)]
    snapshots = [(x, bytes_of(x)) for x in [q, k, v, do, dht]]
    h, _ = module.chunk_fwd_h_fn(k, v, 64, None, False)
    dh0 = allocate((1, 1, 2, 2), torch.float32, q.device)
    dh = module.chunk_bwd_dh_fn(do, q, k, v, 64, 0.5, dht, dh0)
    target = dht.data.double()
    for t in range(1, -1, -1):
        target = (1 - 2. ** -5) * (target + 0.5 * q.data[:, :, t, :, None].double() * do.data[:, :, t, None, :].double())
    torch.testing.assert_close(dh0.data.double(), target, rtol=5e-4, atol=3e-6)
    def reject_helper(run):
        nonlocal rejected
        count = len(launches)
        try: run()
        except (TypeError, ValueError): rejected += 1
        else: raise AssertionError("expected helper rejection")
        assert len(launches) == count
    for alias in [q, k, v, do, dht]:
        reject_helper(lambda: module.chunk_bwd_dh_fn(do, q, k, v, 64, 0.5, dht, alias))
    # Disjoint active views still share caller-owned backing and must reject.
    backing = torch.randn(8)
    aliased_q = Tensor(backing[:4].reshape(1, 1, 2, 2))
    aliased_dh0 = Tensor(backing[4:].reshape(1, 1, 2, 2))
    reject_helper(lambda: module.chunk_bwd_dh_fn(do, aliased_q, k, v, 64, 0.5, dht, aliased_dh0))
    reject_helper(lambda: module.chunk_fwd_h_fn(k, v, 64.0, None, False))
    reject_helper(lambda: module.chunk_fwd_o_fn(h, q, k, v, 64.0, 0.5))
    reject_helper(lambda: module.chunk_bwd_dh_fn(do, q, k, v, 64.0, 0.5, dht, dh0))
    reject_helper(lambda: module.chunk_bwd_dqkv_fn(Tensor(do.data.expand(2, 1, 2, 2)), q, k, v, h, dh, 0.5))
    for x, before in snapshots: assert torch.equal(bytes_of(x), before)
    for _, backing in Tensor.allocated:
        assert bool((backing[:8] == -444.5).all()) and bool((backing[-8:] == -444.5).all())
    print(f"PASS: {len(cases)} full forward/backward cases with repeated backward; {rejected} rejects; {len(launches)} source launches")
    print("PASS: dense independent recurrence and all four adjoints; original backing bytes and allocation canaries")
    print(f"torch={torch.__version__}; Triton={triton.__version__}; CPU interpreter only, no GPU numerical claim")


if __name__ == "__main__": check()

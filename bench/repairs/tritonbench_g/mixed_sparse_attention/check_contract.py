"""CPU interpreter checks; approximate Real comparisons are diagnostics only."""
import os
os.environ["TRITON_INTERPRET"] = "1"
import ast
import contextlib
import importlib.util
import hashlib
import json
from pathlib import Path
import torch


class CudaFacade(torch.Tensor):
    @property
    def is_cuda(self):
        return True


class NonStridedFacade(CudaFacade):
    @property
    def layout(self):
        return torch.sparse_coo


def f(x):
    return x.as_subclass(CudaFacade)


torch.cuda.device = lambda _: contextlib.nullcontext()
torch.manual_seed(571)
path = Path(__file__).with_name("mixed_sparse_attention_repaired.py")
spec = importlib.util.spec_from_file_location("mixed_sparse_checked", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
source = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_triton_mixed_sparse_attn_fwd_kernel")
assert hashlib.sha256(ast.dump(source, include_attributes=False).encode()).hexdigest() == "f5eaff450d5cb5008e79e9e564b320284f34f108a048a48396ec33e332922a35"
count = 1


def case(dim=16, length=65, batch=2, ns=4, nv=65):
    heads = 2
    nr = (length + 63) // 64
    q, k, v = [f(torch.randn(batch, heads, length, dim) * 0.2) for _ in range(3)]
    seq = f(torch.tensor([max(length - b, 0) for b in range(batch)], dtype=torch.int32))
    bc = f(torch.full((batch, heads, nr), ns, dtype=torch.int32))
    cc = f(torch.full((batch, heads, nr), nv, dtype=torch.int32))
    bo = f(torch.zeros(batch, heads, nr, ns, dtype=torch.int32))
    ci = f(torch.zeros(batch, heads, nr, nv, dtype=torch.int32))
    for b in range(batch):
        if int(seq[b]) == 0:
            bc[b].zero_()
            cc[b].zero_()
        else:
            for j in range(ns):
                bo[b, :, :, j] = j * 13 % int(seq[b])
            if nv:
                ci[b] = (torch.arange(nv, dtype=torch.int32) % int(seq[b])).view(1, 1, nv)
    return [q, k, v, seq, bc, bo, cc, ci, 0.25]


def run(a):
    global count
    before = [x.clone() for x in a[:8]]
    out = mod._triton_mixed_sparse_attention(*a)
    q, k, v, seq, bc, bo, cc, ci = a[:8]
    ref = torch.zeros_like(out)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            for row in range(int(seq[b])):
                rp = row // 64
                keys = [int(bo[b, h, rp, j]) + lane for j in range(int(bc[b, h, rp]))
                        for lane in range(64) if int(bo[b, h, rp, j]) + lane <= row]
                keys += [int(ci[b, h, rp, j]) for j in range(int(cc[b, h, rp]))]
                if keys:
                    scores = (q[b, h, row].double() * k[b, h, keys].double()).sum(-1) * a[8]
                    ref[b, h, row] = (scores.softmax(0)[:, None] * v[b, h, keys].double()).sum(0).float()
    torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3, equal_nan=True)
    assert out.shape == q.shape and out.dtype == q.dtype and out.device == q.device
    torch.testing.assert_close(out, out.half().float(), atol=0, rtol=0, equal_nan=True)
    for old, x in zip(before, a[:8]):
        assert torch.equal(x, old)
        if x.numel() and out.numel():
            assert out.data_ptr() != x.data_ptr()
    count += 1


for dim in [16, 32, 64, 128]:
    for length in [0, 1, 63, 64, 65, 129]:
        run(case(dim, length))
    run(case(dim, 65, batch=0))
    run(case(dim, 65, ns=0, nv=0))
    run(case(dim, 65, ns=0, nv=65))
    run(case(dim, 65, ns=4, nv=0))

a = case(length=65)
a[3].zero_()
a[4].zero_()
a[6].zero_()
run(a)
a = case(length=65)
a[2] = a[1]
run(a)
a = case(length=65)
a[0].zero_()
a[3] = a[0].reshape(-1)[:a[3].numel()].view(torch.int32)
a[4].zero_()
a[6].zero_()
run(a)


def reject(change):
    global count
    a = case()
    change(a)
    try:
        mod._triton_mixed_sparse_attention(*a)
    except ValueError:
        count += 1
    else:
        raise AssertionError("invalid call accepted")


reject(lambda a: a.__setitem__(0, a[0].as_subclass(torch.Tensor)))
reject(lambda a: a.__setitem__(0, a[0].as_subclass(NonStridedFacade)))
reject(lambda a: a.__setitem__(3, a[3].as_subclass(NonStridedFacade)))
reject(lambda a: a.__setitem__(0, a[0].half()))
reject(lambda a: a.__setitem__(0, torch._neg_view(a[0])))
reject(lambda a: a.__setitem__(0, a[0].transpose(2, 3)))
reject(lambda a: a.__setitem__(3, a[3].long()))
reject(lambda a: a.__setitem__(4, a[4][..., :1]))
reject(lambda a: a[3].__setitem__(0, -1))
reject(lambda a: a[3].__setitem__(0, 66))
reject(lambda a: a[4].__setitem__((0, 0, 0), -1))
reject(lambda a: a[4].__setitem__((0, 0, 0), 5))
reject(lambda a: a[6].__setitem__((0, 0, 0), -1))
reject(lambda a: a[6].__setitem__((0, 0, 0), 66))
reject(lambda a: a[5].__setitem__((0, 0, 0, 0), -1))
reject(lambda a: a[5].__setitem__((0, 0, 0, 0), 65))
reject(lambda a: a[7].__setitem__((0, 0, 0, 0), -1))
reject(lambda a: a[7].__setitem__((0, 0, 0, 0), 65))
reject(lambda a: a.__setitem__(8, float("inf")))
reject(lambda a: a.extend([32, 64]))


def unaligned(a):
    shape = a[0].shape
    storage = torch.UntypedStorage(a[0].numel() * 4 + 1)
    a[0] = f(torch.empty(0, dtype=torch.float32).set_(storage[1:], 0, shape))


reject(unaligned)
print(json.dumps({"cases": count, "status": "pass", "backend":
    "CPU interpreter/CUDA metadata facade; Real comparison tolerance is diagnostic, not GPU numerical validation"}))

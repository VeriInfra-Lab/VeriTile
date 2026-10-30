"""CPU interpreter/guard checks, not GPU numerical instruction validation."""
import os
os.environ["TRITON_INTERPRET"] = "1"
import contextlib
import importlib.util
import json
import ast
import hashlib
from pathlib import Path
import torch


class CudaFacade(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def f(x):
    return x.as_subclass(CudaFacade)


torch.cuda.device = lambda _: contextlib.nullcontext()
torch.manual_seed(571)
path = Path(__file__).with_name("block_sparse_attn_repaired.py")
spec = importlib.util.spec_from_file_location("block_sparse_checked", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
source = next(n for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "block_sparse_attention_kernel")
assert hashlib.sha256(ast.dump(source, include_attributes=False).encode()).hexdigest() == \
    "3d61b6f42ebad945a3b40313608a63e37ef669ecffc9a55217eb6398d1e4f9d1"
count = 1


def case(blocks=3, length=33, batch=2, even=False, duplicate=False):
    heads, kv = 4, 2
    padded = ((length + 15) // 16) * 16 if even else length
    q = f(torch.randn(batch, heads, padded, 16 * blocks) * 0.2)
    k = f(torch.randn(batch, kv, padded, 16 * blocks) * 0.2)
    v = f(torch.randn(batch, kv, padded, 16 * blocks))
    nr = (length + 15) // 16
    rows, cols = [0], []
    for row in range(nr):
        cols.extend(range(row + 1))
        if duplicate:
            cols.append(row)
        rows.append(len(cols))
    return [q, k, v, f(torch.tensor(rows, dtype=torch.int32)),
            f(torch.tensor(cols or [0], dtype=torch.int32)), nr + 1,
            max(len(cols), 1), 1, 0.25, heads, kv, length, 16, even, 16, even, 16, blocks]


def run(a):
    global count
    before = [x.clone() for x in a[:5]]
    out = mod.block_sparse_attention(*a)
    q, k, v, rows, cols = a[:5]
    ref = torch.zeros_like(out)
    for b in range(q.shape[0]):
        for h in range(a[9]):
            for r in range(a[11]):
                rp = r // 16
                visits = [int(cols[j]) for j in range(int(rows[rp]), int(rows[rp + 1]))]
                if any(col * 16 > r for col in visits):
                    ref[b, h, r] = float("nan")
                    continue
                keys = [col * 16 + j for col in visits for j in range(16) if col * 16 + j <= r]
                if keys:
                    kh = h // (a[9] // a[10])
                    scores = (q[b, h, r].double() * k[b, kh, keys].double()).sum(-1) * a[8]
                    ref[b, h, r] = (scores.softmax(0)[:, None] * v[b, kh, keys].double()).sum(0).float()
    torch.testing.assert_close(out, ref, atol=8e-5, rtol=4e-5, equal_nan=True)
    assert out.shape == (q.shape[0], a[9], a[11], q.shape[-1])
    assert out.dtype == torch.float32 and out.device == q.device and out.is_contiguous()
    for x, old in zip(a[:5], before):
        assert torch.equal(x, old)
        if x.numel() and out.numel():
            assert out.data_ptr() != x.data_ptr()
    count += 1


for blocks in [1, 2, 3, 4]:
    for length in [1, 16, 17, 33]:
        run(case(blocks, length))
    run(case(blocks, 17, even=True, duplicate=True))
    run(case(blocks, 0))
    run(case(blocks, 33, batch=0))

a = case(length=17)
a[3].zero_()
run(a)
a = case(length=17)
a[3][:] = f(torch.tensor([2, 1, 0], dtype=torch.int32))
run(a)
a = case(length=17)
a[3][:] = f(torch.tensor([0, 1, 1], dtype=torch.int32))
a[4][0] = 1
run(a)  # retained all-masked NaN, followed by an empty zero row
a = case(blocks=3, length=17)
a[2] = a[1]  # read/read alias remains valid
run(a)
a = case(blocks=3, length=17)
a[0].zero_()
a[3] = a[0].reshape(-1)[:a[3].numel()].view(torch.int32)
run(a)  # mixed raw-word read alias; all CSR rows empty


def reject(change):
    global count
    a = case()
    change(a)
    try:
        mod.block_sparse_attention(*a)
    except ValueError:
        count += 1
    else:
        raise AssertionError("invalid call accepted")


reject(lambda a: a.__setitem__(0, a[0].as_subclass(torch.Tensor)))
reject(lambda a: a.__setitem__(0, a[0].double()))
reject(lambda a: a.__setitem__(0, torch._neg_view(a[0])))
reject(lambda a: a.__setitem__(0, a[0].transpose(2, 3)))
reject(lambda a: a.__setitem__(3, a[3].long()))
reject(lambda a: a[3].__setitem__(0, -1))
reject(lambda a: a[3].__setitem__(0, a[6] + 1))
reject(lambda a: a[4].__setitem__(0, -1))
reject(lambda a: a[4].__setitem__(0, 3))
reject(lambda a: a.__setitem__(6, a[6] + 1))
reject(lambda a: a.__setitem__(5, 2**31))
reject(lambda a: a.__setitem__(8, float("inf")))
reject(lambda a: a.__setitem__(9, 3))
reject(lambda a: a.__setitem__(11, -1))
reject(lambda a: a.__setitem__(12, 32))
reject(lambda a: a.__setitem__(13, True))
reject(lambda a: a.__setitem__(15, True))
reject(lambda a: a.__setitem__(17, 2))


def unaligned(a):
    shape = a[0].shape
    storage = torch.UntypedStorage(a[0].numel() * 4 + 1)
    a[0] = f(torch.empty(0, dtype=torch.float32).set_(storage[1:], 0, shape))


reject(unaligned)
print(json.dumps({"cases": count, "status": "pass",
                  "backend": "CPU Triton interpreter and CUDA header facade; no GPU numerical validation"}))

"""Targeted actual-source CPU interpreter checks; no CUDA execution is claimed."""
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

os.environ["TRITON_INTERPRET"] = "1"
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("repair", HERE / "kv_cache_filling_repaired.py")
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class Tensor:
    """CPU payload with explicitly mocked CUDA metadata for host replay."""
    def __init__(self, value, **overrides):
        self.value = value
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        if name == "is_cuda":
            return True
        if name == "device":
            return torch.device("cuda:0")
        return getattr(self.value, name)


class Launch:
    count = 0

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            Launch.count += 1
            kernel[grid](*[x.value if isinstance(x, Tensor) else x for x in args], **kwargs)
        return invoke


kernel = repair._fill_rows
repair._fill_rows = Launch()
repair.torch = SimpleNamespace(
    Tensor=Tensor, strided=torch.strided, float32=torch.float32,
    uint8=torch.uint8, int32=torch.int32, int64=torch.int64,
    tensor=lambda values, dtype, device: Tensor(torch.tensor(values, dtype=dtype)),
    cuda=SimpleNamespace(device=lambda _: nullcontext()))


def case(policy, constant=False, empty=False):
    factor = 2 if policy == 4 else 1
    dk, dv, heads, block, pages = 3*factor, 5*factor, 2, 4, 3
    n = 0 if empty else 3
    k = torch.arange(n*heads*dk, dtype=torch.float32).reshape(n,heads,dk)
    v = torch.arange(n*heads*dv, dtype=torch.float32).reshape(n,heads,dv).neg()
    if policy == 0 and not constant and not empty:
        words = torch.tensor([0,-2147483648,1,2139095040,2143289345,-1],dtype=torch.int32)
        k.copy_(words.repeat((k.numel()+5)//6)[:k.numel()].view(torch.float32).reshape(k.shape))
    if constant:
        k.fill_(2)
        v.fill_(-3)
    dtype = torch.float32 if policy == 0 else torch.uint8
    kb = torch.full((pages+2,block,heads,3), 73, dtype=dtype)
    vb = torch.full((pages+2,block,heads,5), 91, dtype=dtype)
    kc, vc = kb[1:-1], vb[1:-1]
    km = torch.full((pages,block,heads,2), 19., dtype=torch.float32)
    vm = torch.full_like(km, 23.)
    q = [0,0] if empty else [2,1]
    starts = torch.tensor([0,0] if empty else [0,2], dtype=torch.int32)
    counts = torch.tensor(q, dtype=torch.int32)
    totals = torch.tensor([0,0] if empty else [5,2], dtype=torch.int32)
    table = torch.tensor([[0,1],[2,0]], dtype=torch.int32)
    tensors = [k,v,kc,vc,starts,counts,totals,table,km,vm]
    before = [x.clone() for x in tensors]
    k_back, v_back = kb.clone(), vb.clone()
    args = [Tensor(x) for x in tensors[:7]]+[2,Tensor(table),Tensor(km),Tensor(vm),policy]
    launches = Launch.count
    assert repair.fill_kv_cache(*args) is None
    assert Launch.count-launches == (0 if empty else 1)
    ek, ev, emk, emv = before[2].clone(), before[3].clone(), before[8].clone(), before[9].clone()
    for b in range(2):
        history = int(totals[b]-counts[b])
        for i in range(q[b]):
            page = int(table[b,(history+i)//block])
            slot = (history+i)%block
            row = int(starts[b])+i
            for source, cache, meta in ((k,ek,emk),(v,ev,emv)):
                for head in range(heads):
                    x = source[row,head]
                    if not policy:
                        cache[page,slot,head] = x
                        continue
                    lo, hi = x.min(), x.max()
                    scale = torch.tensor(1.) if lo == hi else (hi-lo)/(15 if policy == 4 else 255)
                    zero = (torch.tensor(0.)-lo)/scale
                    code = (x/scale+zero+0.5).to(torch.uint8)
                    if policy == 4:
                        half = len(code)//2
                        code = code[:half]+code[half:]*16
                    cache[page,slot,head] = code
                    meta[page,slot,head] = torch.stack((scale,zero))
    for actual, expected in ((kc,ek),(vc,ev),(km,emk),(vm,emv)):
        assert equal(actual,expected), (policy,constant,actual,expected)
    for i in (0,1,4,5,6,7):
        assert equal(tensors[i],before[i])
    assert torch.equal(kb[[0,-1]],k_back[[0,-1]])
    assert torch.equal(vb[[0,-1]],v_back[[0,-1]])
    return args


def equal(a,b):
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32),b.view(torch.int32))
    return torch.equal(a,b)


def rejected(args, index, replacement):
    args = list(args)
    args[index] = replacement
    before = Launch.count
    try:
        repair.fill_kv_cache(*args)
    except (ValueError, TypeError):
        assert Launch.count == before
    else:
        raise AssertionError("invalid input admitted")


for policy in (0,4,8):
    args = case(policy)
    case(policy,constant=True)
    case(policy,empty=True)
    rejected(args,4,Tensor(torch.tensor([-1,2],dtype=torch.int32)))
    rejected(args,5,Tensor(torch.tensor([-1,1],dtype=torch.int32)))
    rejected(args,6,Tensor(torch.tensor([1,2],dtype=torch.int32)))
    rejected(args,8,Tensor(torch.tensor([[-1,1],[2,0]],dtype=torch.int32)))
    colliding = list(args)
    colliding[6] = Tensor(torch.tensor([5,1],dtype=torch.int32))
    rejected(colliding,8,Tensor(torch.tensor([[0,0],[0,0]],dtype=torch.int32)))
    rejected(args,0,Tensor(args[0].value,is_cuda=False))
    rejected(args,0,Tensor(args[0].value.to(torch.float64)))
    rejected(args,1,Tensor(args[1].value,device=torch.device("cuda:1")))
    rejected(args,0,Tensor(args[0].value.transpose(0,1)))
    rejected(args,0,Tensor(torch._neg_view(args[0].value)))
print("KV source interpreter: copy/4-bit/8-bit, padding, constants, history, offsets, frame, empty and rejection checks passed")

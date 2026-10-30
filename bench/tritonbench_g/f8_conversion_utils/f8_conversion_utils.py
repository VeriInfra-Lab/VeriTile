import torch
import triton
import triton.language as tl

CHUNK_SIZE = 1 << 30

@triton.jit
def kernel_f8_to_f16(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)
    tl.store(Y + offs, x, mask=mask)

@triton.jit
def kernel_f16_to_f8(Y, X, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    tl.store(Y + offs, x, mask=mask)


def _check_input(x, dtypes, *, decode):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.layout != torch.strided:
        raise ValueError("x must use strided layout")
    if decode:
        if x.dtype != torch.int8:
            raise TypeError("decode input must contain packed int8 FP8 bytes")
    elif x.dtype not in (torch.float16, torch.float32):
        raise TypeError("encode input must be FP16 or FP32")
    if dtypes is not tl.float8e5 and dtypes is not tl.float8e4nv:
        raise TypeError("dtypes must be tl.float8e5 or tl.float8e4nv")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.is_neg():
        raise ValueError("x must not carry a lazy negative flag")
    if x.numel() and x.data_ptr() % x.element_size():
        raise ValueError("x must have natural element alignment")


def f8_to_f16_repaired(x, dtypes=tl.float8e5):
    _check_input(x, dtypes, decode=True)
    with torch.cuda.device(x.device):
        ret = torch.empty_like(x, dtype=torch.float16)
        n = x.numel()
        if n == 0:
            return ret
        source, target = x.view(-1), ret.view(-1)
        for start in range(0, n, CHUNK_SIZE):
            count = min(CHUNK_SIZE, n - start)
            grid = (triton.cdiv(count, 1024),)
            kernel_f8_to_f16[grid](target[start:start+count],
                triton.reinterpret(source[start:start+count], dtypes), count, BLOCK_SIZE=1024)
        return ret


def f16_to_f8_repaired(x, dtypes=tl.float8e5):
    _check_input(x, dtypes, decode=False)
    with torch.cuda.device(x.device):
        ret = torch.empty_like(x, dtype=torch.int8)
        n = x.numel()
        if n == 0:
            return ret
        source, target = x.view(-1), ret.view(-1)
        for start in range(0, n, CHUNK_SIZE):
            count = min(CHUNK_SIZE, n - start)
            grid = (triton.cdiv(count, 1024),)
            kernel_f16_to_f8[grid](triton.reinterpret(target[start:start+count], dtypes),
                source[start:start+count], count, BLOCK_SIZE=1024)
        return ret

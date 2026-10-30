# Candidate: strength-reduce both K-tail masks to a scalar remaining length.
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(c_ptr, a_ptr, b_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = tl.cast(accumulator, tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

def _validate(c, a, b, M, N, K, BM, BN, BK):
    if any(type(d) is not int or d < 0 for d in (M, N, K)):
        raise ValueError("dimensions must be nonnegative integers")
    if any(type(d) is not int or d < 16 or d & (d - 1) for d in (BM, BN, BK)):
        raise ValueError("blocks must be legal powers of two at least 16")
    for x, shape in ((a, (M, K)), (b, (K, N)), (c, (M, N))):
        if (not isinstance(x, torch.Tensor) or not x.is_cuda or x.dtype != torch.float32
                or x.layout != torch.strided or x.ndim != 2 or tuple(x.shape) != shape
                or x.is_neg() or x.is_conj() or any(s <= 0 for s in x.stride())):
            raise ValueError("expected ordinary positive-stride FP32 CUDA matrices")
        if x.numel() and x.data_ptr() % 4:
            raise ValueError("nonempty FP32 matrices must have four-byte aligned data pointers")
        rows, cols = shape
        sr, sc = x.stride()
        if rows and cols and not (sr >= cols * sc or sc >= rows * sr):
            raise ValueError("matrix views must have separated rows or columns")
    if a.device != b.device or a.device != c.device:
        raise ValueError("all matrices must share one device")
    if c.numel() and any(x.numel() and c.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
                         for x in (a, b)):
        raise ValueError("output storage must be independent from input storage")
    if M == 0 or N == 0:
        return
    pm, pn, pk = triton.cdiv(M, BM) * BM, triton.cdiv(N, BN) * BN, triton.cdiv(K, BK) * BK
    if (max(M, N, K, BM, BN, BK, pm, pn, pk + BK, *a.stride(), *b.stride(), *c.stride()) >= 2**31
            or (M - 1) * a.stride(0) + (pk + BK) * a.stride(1) >= 2**31
            or (pk + BK) * b.stride(0) + (N - 1) * b.stride(1) >= 2**31
            or pm * c.stride(0) + pn * c.stride(1) >= 2**31):
        raise ValueError("source formed addresses exceed the admitted signed32 domain")


def matmul(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K):
    _validate(c, a, b, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
    if M == 0 or N == 0:
        return
    with torch.cuda.device(a.device):
        matmul_kernel[triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N)](
            c, a, b, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
        )

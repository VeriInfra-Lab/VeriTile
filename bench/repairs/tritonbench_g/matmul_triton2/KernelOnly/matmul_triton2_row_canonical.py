# Candidate: canonicalize the exact uint64 row-block product with the static tile width first.
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    M = tl.cast(M, tl.uint64)
    N = tl.cast(N, tl.uint64)
    K = tl.cast(K, tl.uint64)
    stride_am = tl.cast(stride_am, tl.uint64)
    stride_ak = tl.cast(stride_ak, tl.uint64)
    stride_bk = tl.cast(stride_bk, tl.uint64)
    stride_bn = tl.cast(stride_bn, tl.uint64)
    stride_cm = tl.cast(stride_cm, tl.uint64)
    stride_cn = tl.cast(stride_cn, tl.uint64)
    pid = tl.program_id(axis=0).to(tl.uint64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.cast(min(num_pid_m - first_pid_m, GROUP_SIZE_M), tl.uint64)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = BLOCK_SIZE_M * pid_m + tl.arange(0, BLOCK_SIZE_M).to(tl.uint64)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.uint64)
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.uint64)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_bn[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + (offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn)
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_matmul(a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("expected Tensor inputs")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError("expected strided inputs")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("expected FP32 inputs")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("expected CUDA inputs")
    if a.device != b.device:
        raise ValueError("expected the same device")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible dimensions")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("expected contiguous inputs")
    if a.is_neg() or b.is_neg():
        raise ValueError("expected resolved input storage")
    M, K = a.shape
    _, N = b.shape
    if M >= 2**63 or K >= 2**63 or N >= 2**63:
        raise ValueError("dimensions must fit signed 64-bit metadata")
    if (a.numel() > 0 and a.data_ptr() % 4 != 0) or (b.numel() > 0 and b.data_ptr() % 4 != 0):
        raise ValueError("nonempty FP32 inputs require aligned pointers")

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if M == 0 or N == 0:
        return c
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    with torch.cuda.device(a.device):
        matmul_kernel[grid](
            a, b, c, M, N, K,
            K, 1,
            N, 1,
            N, 1
        )
    return c

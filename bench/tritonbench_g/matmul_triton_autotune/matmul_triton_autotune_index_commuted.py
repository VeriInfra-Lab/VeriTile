"""Integer-index candidate derived from repaired matmul_triton_autotune.py.

Only the grouped tile-count product is commuted. This is a newly introduced
candidate source, not a historical original. It imports the unchanged original activation helper;
the caller supplies one of the retained CUDA/HIP configurations.
"""
import triton
import triton.language as tl
from matmul_triton_autotune import leaky_relu

@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,  
        stride_bk, stride_bn,  
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  
        GROUP_SIZE_M: tl.constexpr,  
        ACTIVATION: tl.constexpr  
):
    """Kernel for computing the matrix multiplication C = A x B.
    A has shape (M, K), B has shape (K, N), and C has shape (M, N).
    """
    # Widen before every address/control product, including specialized strides.
    M, N, K = tl.cast(M, tl.int64), tl.cast(N, tl.int64), tl.cast(K, tl.int64)
    stride_am, stride_ak = tl.cast(stride_am, tl.int64), tl.cast(stride_ak, tl.int64)
    stride_bk, stride_bn = tl.cast(stride_bk, tl.int64), tl.cast(stride_bn, tl.int64)
    stride_cm, stride_cn = tl.cast(stride_cm, tl.int64), tl.cast(stride_cn, tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_n * GROUP_SIZE_M
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        offs_kk = k * BLOCK_SIZE_K + offs_k
        # Inactive lanes form valid pointers as well; no final padded advance.
        safe_k = tl.where(offs_kk < K, offs_kk, 0)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + safe_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (safe_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        a = tl.load(a_ptrs, mask=offs_kk[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_kk[:, None] < K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    safe_cm = tl.where(offs_cm < M, offs_cm, 0)
    safe_cn = tl.where(offs_cn < N, offs_cn, 0)
    c_ptrs = c_ptr + stride_cm * safe_cm[:, None] + stride_cn * safe_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

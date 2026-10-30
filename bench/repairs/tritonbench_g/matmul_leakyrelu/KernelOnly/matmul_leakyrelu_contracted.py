# Existing contracted DSL profile expressed as a complete source candidate.
import torch
import triton
import triton.language as tl

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
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply activation function if specified
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    
    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, x * 0.01)

def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    if c.data_ptr() + 2 * (triton.cdiv(M, 32) * 32 * max(N, 1) + triton.cdiv(N, 32) * 32) >= 2**64:
        raise ValueError("formed output byte pointers exceed uint64")
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    with torch.cuda.device(a.device):
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32,
            GROUP_SIZE_M=4,
            ACTIVATION=activation
        )
    return c

_unchecked_matmul = matmul


def matmul(a, b, activation=""):
    """Checked FP32 input contract; the original kernel and FP16 output are retained."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be tensors")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError("a and b must have strided layout")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must have rank two")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("a and b must be FP32")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible dimensions")
    if not a.is_contiguous():
        raise ValueError("a must be contiguous")
    if a.is_neg() or b.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if type(activation) is not str:
        raise TypeError("activation must be a string")
    if any(x.numel() and x.data_ptr() % 4 for x in (a, b)):
        raise ValueError("nonempty FP32 inputs must have four-byte aligned data pointers")
    m, k = a.shape
    _, n = b.shape
    am, ak = a.stride()
    bk, bn = b.stride()
    limit = 2**31
    if min(am, ak, bk, bn) < 0:
        raise ValueError("negative strides are unsupported")
    rows = triton.cdiv(m, 32)
    cols = triton.cdiv(n, 32)
    padded_k = triton.cdiv(k, 32) * 32
    if (m + 32 >= limit or n + 32 >= limit or k + 32 >= limit
            or max(am, ak, bk, bn) >= limit
            or rows * cols >= limit or 4 * cols >= limit
            or m * am + (padded_k + 32) * ak >= limit
            or (padded_k + 32) * bk + n * bn >= limit
            or rows * 32 * max(n, 1) + cols * 32 >= limit):
        raise ValueError("formed indices exceed the source int32 domain")
    if (a.data_ptr() + 4 * (m * am + (padded_k + 32) * ak) >= 2**64
            or b.data_ptr() + 4 * ((padded_k + 32) * bk + n * bn) >= 2**64):
        raise ValueError("formed input byte pointers exceed uint64")
    return _unchecked_matmul(a, b, activation)

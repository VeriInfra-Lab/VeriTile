
import torch
import triton
import triton.language as tl

# Autotune configuration functions
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # Other configurations omitted for brevity
    ]

def get_hip_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
        # Other configurations omitted for brevity
    ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

# Triton kernel for matrix multiplication with autotuning
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
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
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
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

@triton.jit
def leaky_relu(x):
    """Applies the Leaky ReLU activation function."""
    return tl.where(x >= 0, x, 0.01 * x)

def matmul_view_safe(a, b, activation=""):
    """Wrapper function for matrix multiplication using Triton.
    a: Tensor with shape (M, K)
    b: Tensor with shape (K, N)
    activation: Activation function to be applied (optional)
    Returns the result of the matrix multiplication.
    """
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    if M == 0 or N == 0:
        return c
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    with torch.cuda.device(a.device):
        matmul_kernel[grid](
            a, b, c,  
            M, N, K,  
            a.stride(0), a.stride(1),  
            b.stride(0), b.stride(1),  
            c.stride(0), c.stride(1),  
            ACTIVATION=activation  
        )
    return c





_unchecked_matmul_view_safe = matmul_view_safe


def matmul_view_safe(a, b, activation=""):
    """Checked raw FP32 views for the original CUDA/HIP configurations."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be tensors")
    if a.layout != torch.strided or b.layout != torch.strided or a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be rank-two strided tensors")
    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise TypeError("a and b must be FP32")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("a and b must share a CUDA/HIP device")
    if a.shape[1] != b.shape[0] or not a.is_contiguous():
        raise ValueError("compatible dimensions and contiguous a are required")
    if a.is_neg() or b.is_neg() or a.is_conj() or b.is_conj():
        raise ValueError("unresolved tensor views are unsupported")
    if min(*a.stride(), *b.stride()) < 0:
        raise ValueError("negative strides are unsupported")
    if any(x.numel() and x.data_ptr() % 4 for x in (a, b)):
        raise ValueError("nonempty FP32 inputs need naturally aligned pointers")
    if type(activation) is not str:
        raise TypeError("activation must be a string")
    return _unchecked_matmul_view_safe(a, b, activation)

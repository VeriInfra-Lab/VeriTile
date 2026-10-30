
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=2,
            num_warps=8,
        ),
        # Additional configurations omitted for brevity
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # Pointers to matrices
    M, N, K,  # Matrix dimensions
    stride_am, stride_ak,  # Strides for matrix A
    stride_bk, stride_bn,  # Strides for matrix B
    stride_cm, stride_cn,  # Strides for matrix C
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    M = M.to(tl.int64)
    N = N.to(tl.int64)
    K = K.to(tl.int64)
    stride_am = stride_am.to(tl.int64)
    stride_ak = stride_ak.to(tl.int64)
    stride_bk = stride_bk.to(tl.int64)
    stride_bn = stride_bn.to(tl.int64)
    stride_cm = stride_cm.to(tl.int64)
    stride_cn = stride_cn.to(tl.int64)
    pid = tl.program_id(axis=0).to(tl.int64)
    num_pid_m = (M // BLOCK_SIZE_M + (M % BLOCK_SIZE_M != 0))
    num_pid_n = (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))
    active_group_m = min(GROUP_SIZE_M, num_pid_m)
    num_pid_in_group = active_group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * active_group_m
    group_size_m = min(num_pid_m - first_pid_m, active_group_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    lanes_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_tiles = K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0)
    for k in range(0, k_tiles):
        offs_k = k * BLOCK_SIZE_K + lanes_k
        safe_k = tl.where(offs_k < K, offs_k, 0)
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + safe_k[None, :] * stride_ak
        b_ptrs = b_ptr + safe_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
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
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert not a.is_neg() and not a.is_conj(), "Matrix A must be resolved"
    assert not b.is_neg() and not b.is_conj(), "Matrix B must be resolved"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    if M == 0 or N == 0:
        return c
    if K == 0:
        c.zero_()
        return c
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c




##################################################################################################################################################


def test_matmul():
    results = {}

    # Test case 1: Basic matrix multiplication without activation
    a = torch.randn((256, 64), device='cuda', dtype=torch.float16)
    b = torch.randn((64, 256), device='cuda', dtype=torch.float16)
    c = matmul(a, b)
    results["test_case_1"] = c

    # Test case 2: Matrix multiplication with leaky ReLU activation
    c_with_activation = matmul(a, b, activation="leaky_relu")
    results["test_case_2"] = c_with_activation

    # Test case 3: Matrix multiplication with larger dimensions
    a_large = torch.randn((512, 128), device='cuda', dtype=torch.float16)
    b_large = torch.randn((128, 512), device='cuda', dtype=torch.float16)
    c_large = matmul(a_large, b_large)
    results["test_case_3"] = c_large

    # Test case 4: Matrix multiplication with larger dimensions and leaky ReLU activation
    c_large_with_activation = matmul(a_large, b_large, activation="leaky_relu")
    results["test_case_4"] = c_large_with_activation

    return results

result_gold = test_matmul()

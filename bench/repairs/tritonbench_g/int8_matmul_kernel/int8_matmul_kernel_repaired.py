
import torch
import triton
import triton.language as tl

def get_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 256,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 8,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "GROUP_SIZE_M": 4,
            },
            num_stages=4,
            num_warps=4,
        ),
    ]

@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    tl.static_assert(
        K % (4 * BLOCK_SIZE_K) == 0,
        "K / 4 must be divisible by BLOCK_SIZE_K => K divisible by 4*BLOCK_SIZE_K",
    )
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for i in range(4):
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K)):
            k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0).to(tl.int8)  # Cast to int8.
            b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0)
            mask = 3 << (2 * i)
            b = ((b_uint8 & mask) >> (2 * i)).to(tl.int8)  # Cast to int8.
            tensor_full = tl.full((1,), 1, dtype=tl.int8)  # Use int8.
            accumulator += tl.dot(a, (b - tensor_full), out_dtype=tl.int32)  # Keep a and b as int8.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
    assert (
        a.shape[1] == b.shape[0] * 4
    ), "Incompatible dimensions, the weight matrix need to be packed"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    if c.data_ptr() + 4 * ((M + 256) * max(N, 1) + N + 256) >= 2**63:
        raise ValueError("formed output byte pointers exceed signed64")
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c

_unchecked_matmul = matmul


def matmul(a, b):
    """Checked FP32-by-packed-uint8 contract for the unchanged autotuned source."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("a and b must be tensors")
    if a.layout != torch.strided or b.layout != torch.strided:
        raise ValueError("a and b must have strided layout")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must have rank two")
    if a.dtype != torch.float32 or b.dtype != torch.uint8:
        raise TypeError("a must be FP32 and b must be packed uint8")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("a and b must be on the same CUDA device")
    if a.device.index != torch.cuda.current_device():
        raise ValueError("the input device must be the current CUDA device")
    if a.shape[1] != 4 * b.shape[0]:
        raise ValueError("incompatible packed dimensions")
    if not a.is_contiguous():
        raise ValueError("a must be contiguous")
    if a.is_neg() or b.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if a.numel() and a.data_ptr() % 4:
        raise ValueError("nonempty FP32 a must have a four-byte aligned data pointer")
    m, k = a.shape
    _, n = b.shape
    am, ak = a.stride()
    bk, bn = b.stride()
    limit = 2**31
    if min(am, ak, bk, bn) < 0:
        raise ValueError("negative strides are unsupported")
    if max(m, n, k) + 512 >= limit or max(am, ak, bk, bn) + 256 >= limit:
        raise ValueError("dimensions and strides exceed the source int32 domain")
    for config in get_autotune_config():
        bm = config.kwargs["BLOCK_SIZE_M"]
        block_n = config.kwargs["BLOCK_SIZE_N"]
        group_m = config.kwargs["GROUP_SIZE_M"]
        rows = triton.cdiv(m, bm)
        cols = triton.cdiv(n, block_n)
        if rows * cols >= limit or group_m * cols >= limit:
            raise ValueError("grouped indices exceed the source int32 domain")
    if (m * am + (k + 128) * ak >= limit
            or (k // 4 + 128) * bk + n * bn >= limit
            or (m + 256) * max(n, 1) + n + 256 >= limit):
        raise ValueError("formed offsets exceed the source int32 domain")
    if (a.data_ptr() + 4 * (m * am + (k + 128) * ak) >= 2**63
            or b.data_ptr() + (k // 4 + 128) * bk + n * bn >= 2**63):
        raise ValueError("formed input byte pointers exceed signed64")
    if not torch.isfinite(a).all().item():
        raise ValueError("FP32-to-int8 conversion requires finite inputs")
    truncated = torch.trunc(a)
    if not ((truncated >= -128) & (truncated <= 127)).all().item():
        raise ValueError("truncated inputs must fit signed int8")
    return _unchecked_matmul(a, b)

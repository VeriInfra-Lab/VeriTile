
import torch
import triton
import triton.language as tl

# global quantize and transpose
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8}, num_warps=4),
        # ...
    ],
    key=["M", "N"],
)
@triton.jit
def _quantize_global_transpose(
    A,
    absmax_inv_ptr,
    B,
    stride_am,
    stride_an,
    stride_bn,
    stride_bm,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    absmax_inv = tl.load(absmax_inv_ptr)

    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    output = tl.extra.cuda.libdevice.llrint(127.0 * (a * absmax_inv))

    tl.store(B, output, mask=mask)

def quantize_global_transpose(input):
    absmax = input.abs().max().unsqueeze(0)
    absmax_inv = 1.0 / absmax
    M, N = input.shape
    out = torch.empty(N, M, device="cuda", dtype=torch.int8)

    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _quantize_global_transpose[grid](
        input,
        absmax_inv,
        out,
        input.stride(0),
        input.stride(1),
        out.stride(0),
        out.stride(1),
        M,
        N,
    )
    return out, absmax

_unchecked_quantize_global_transpose = quantize_global_transpose


def quantize_global_transpose(input):
    """FP32 resolved-view contract with checked source integer envelopes."""
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a tensor")
    if input.layout != torch.strided:
        raise ValueError("input must have strided layout")
    if input.ndim != 2:
        raise ValueError("input must have rank two")
    if input.dtype != torch.float32:
        raise TypeError("input must be FP32")
    if not input.is_cuda or input.device.index != torch.cuda.current_device():
        raise ValueError("input must be on the current CUDA device")
    if input.is_neg():
        raise ValueError("unresolved negative views are unsupported")
    if input.numel() and input.data_ptr() % 4:
        raise ValueError("nonempty FP32 input must have a four-byte aligned data pointer")
    m, n = input.shape
    sam, san = input.stride()
    if sam < 0 or san < 0:
        raise ValueError("negative strides are unsupported")
    if sam != 1 and san != 1:
        raise ValueError("one input stride must be one")
    gm, gn = triton.cdiv(m, 128), triton.cdiv(n, 128)
    limit = 2**31
    if (m <= 0 or n <= 0 or m + 128 >= limit or n + 128 >= limit
            or sam >= limit or san >= limit or gm * gn >= limit or 8 * gn >= limit
            or gm * 128 * sam + gn * 128 * san >= limit
            or gm * 128 + gn * 128 * m >= limit):
        raise ValueError("formed indices exceed the source int32 domain")
    return _unchecked_quantize_global_transpose(input)

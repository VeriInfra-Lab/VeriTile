
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['K', 'N'],
)


@triton.jit
def dequantize_kernel(
    b_ptr, b_scale_ptr, fpb_ptr,
    K, N,
    stride_bk, stride_bn, stride_scale,
    stride_fpbk, stride_fpbn,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    k_block_idx = tl.program_id(axis=0).to(tl.int64)
    n_block_idx = tl.program_id(axis=1).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    b_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_bk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_bn
    fpb_offs = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None]) * stride_fpbk + \
        (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_fpbn
    bs_offs = (n_block_idx * BLOCK_SIZE_N + offs_n[None, :]) * stride_scale
    n_mask = n_block_idx * BLOCK_SIZE_N + offs_n[None, :] < N
    mask = (k_block_idx * BLOCK_SIZE_K + offs_k[:, None] < K) & n_mask
    int_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
    scale_b = tl.load(b_scale_ptr + bs_offs, mask=n_mask, other=0.0)
    tl.store(fpb_ptr + fpb_offs, int_b * scale_b, mask=mask)

def _check(a, b, b_scale, out):
    if not all(isinstance(x, torch.Tensor) for x in (a, b, b_scale)):
        raise TypeError("inputs must be tensors")
    if a.ndim != 2 or b.ndim != 2 or b_scale.ndim != 1:
        raise ValueError("expected matrices and column scales")
    if any(x.layout != torch.strided for x in (a, b, b_scale)):
        raise TypeError("inputs must have strided layout")
    if a.dtype != torch.float32 or b.dtype != torch.int8 or b_scale.dtype != torch.float32:
        raise TypeError("expected FP32 A/scales and signed-int8 B")
    if not a.is_cuda or a.device != b.device or a.device != b_scale.device:
        raise ValueError("inputs must share one CUDA device")
    if any(not x.is_contiguous() or x.is_neg() for x in (a, b, b_scale)):
        raise ValueError("inputs must be contiguous with resolved negative bits")
    if torch._C._overlaps(a, b) or torch._C._overlaps(b_scale, b):
        raise ValueError("INT8 and FP32 inputs must not share raw storage")
    M, K = a.shape
    _, N = b.shape
    if K != b.shape[0] or b_scale.shape != (N,):
        raise ValueError("incompatible matrix or scale shape")
    if a.numel() >= 2**61 or b.numel() >= 2**61 or M * N >= 2**61:
        raise ValueError("FP32 input, scratch and output byte extents must fit signed int64")
    if K != 0 and N != 0 and (K >= 2**31 or N > 65535 * 32):
        raise ValueError("dequantization dimensions exceed the supported launch bounds")
    if any(x.numel() > 0 and x.data_ptr() % 4 != 0 for x in (a, b_scale)):
        raise ValueError("FP32 inputs must be aligned to four bytes")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a tensor or None")
        if out.layout != torch.strided or out.shape != (M, N) or out.dtype != torch.float32 or out.device != a.device:
            raise ValueError("incompatible output descriptor")
        if not out.is_contiguous() or out.is_neg():
            raise ValueError("output must be contiguous with a resolved negative bit")
        if any(torch._C._overlaps(out, x) for x in (a, b, b_scale)):
            raise ValueError("output must not share storage with an input")
        if out.numel() > 0 and out.data_ptr() % 4 != 0:
            raise ValueError("FP32 output must be aligned to four bytes")
    return M, K, N


def matmul_dequantize_int8(a, b, b_scale, out=None):
    M, K, N = _check(a, b, b_scale, out)
    if out is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        c = out
    fp_b = torch.empty((K, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    if K != 0 and N != 0:
        with torch.cuda.device(a.device):
            # Checked contiguity gives these active offsets even for singleton
            # views with unusual inert strides; no input materialization.
            dequantize_kernel[grid](b, b_scale, fp_b, K, N, N, 1, 1, N, 1)
    torch.mm(a, fp_b, out=c)
    return c

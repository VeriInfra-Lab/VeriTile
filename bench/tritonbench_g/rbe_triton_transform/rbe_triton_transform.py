import torch

import triton
import triton.language as tl




@triton.jit
def get_freq_multi_tokens(offs_cn, starting_idx, theta: tl.constexpr, NB_TOKENS: tl.constexpr):
    DIM: tl.constexpr = 128  # in model, dim = self.params.dim // self.params.n_heads
    freqs = offs_cn % DIM
    freqs = freqs.to(tl.float32) / DIM
    freqs = tl.extra.cuda.libdevice.pow(theta, freqs)
    freqs = (tl.arange(0, NB_TOKENS).to(tl.int64) + tl.cast(starting_idx, tl.int64))[:, None] / freqs[None, :]
    return tl.cos(freqs), tl.sin(freqs)


@triton.jit
def rbe_triton(x_ptr, out_ptr,
               M, K,
               stride_x_batch, stride_x_m, stride_x_n,
               stride_out_batch, stride_out_m, stride_out_n,
               start_token_position,
               THETA: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    M = tl.cast(M, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_m = tl.cast(stride_out_m, tl.int64)
    stride_out_n = tl.cast(stride_out_n, tl.int64)
    start_token_position = tl.cast(start_token_position, tl.int64)
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1).to(tl.int64)
    pid_m = pid // tl.cdiv(K, BLOCK_SIZE_K)
    pid_n = pid % tl.cdiv(K, BLOCK_SIZE_K)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64) * 2
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + stride_x_m * offs_m[:, None] + stride_x_n * offs_n[None, :])
    x_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    real = tl.load(x_ptrs, mask=x_real_mask, other=0.0)
    x_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    imag = tl.load(x_ptrs + stride_x_n, mask=x_imag_mask, other=0.0)
    tl.debug_barrier()
    start_block = start_token_position + pid_m * BLOCK_SIZE_M
    cos, sin = get_freq_multi_tokens(offs_cn=offs_n, starting_idx=start_block, theta=THETA, NB_TOKENS=BLOCK_SIZE_M)

    out_real = real * cos - imag * sin
    out_imag = real * sin + imag * cos
    tl.debug_barrier()
    out_ptrs = out_ptr + (
            pid_batch * stride_out_batch + stride_out_m * offs_m[:, None] + stride_out_n * offs_n[None, :])
    out_real_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(out_ptrs, out_real, mask=out_real_mask)
    out_imag_mask = (offs_m[:, None] < M) & (1 + offs_n[None, :] < K)
    tl.store(out_ptrs + stride_out_n, out_imag, mask=out_imag_mask)


def rbe_triton_wrapper(x: torch.Tensor, pos: int) -> torch.Tensor:
    """Rotate ordinary contiguous FP32 logical pairs, including odd tails.

    Input has rank three, no lazy negative metadata, and a signed int64
    position whose padded row positions also fit int64.
    Output has the same shape/dtype/device and independent contiguous storage.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a Tensor")
    if x.layout != torch.strided or x.dtype != torch.float32:
        raise TypeError("expected a strided FP32 Tensor")
    if x.device.type != "cuda":
        raise ValueError("input must be on CUDA")
    if x.ndim != 3 or not x.is_contiguous() or x.is_neg():
        raise ValueError("input must be a resolved contiguous rank-three Tensor")
    if x.numel() and x.data_ptr() % 4:
        raise ValueError("FP32 data pointer must be four-byte aligned")
    if not isinstance(pos, int) or not -(1 << 63) <= pos < (1 << 63):
        raise ValueError("position must be a signed int64 Python integer")
    batch, M, K = x.shape
    if x.numel() and pos + 2 * triton.cdiv(M, 2) - 1 >= (1 << 63):
        raise ValueError("padded row positions must fit signed int64")
    if x.numel() and batch > (1 << 31) - 1:
        raise ValueError("batch dimension exceeds CUDA grid-x limit")
    if x.numel() and triton.cdiv(M, 2) * triton.cdiv(K, 1024) > 65535:
        raise ValueError("launch grid-y exceeds CUDA grid-y limit")
    with torch.cuda.device(x.device):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        if x.numel() == 0:
            return out
        grid = lambda META: (
            batch, triton.cdiv(META["M"], META["BLOCK_SIZE_M"]) * triton.cdiv(META["K"], META["BLOCK_SIZE_K"]),)
        rbe_triton[grid](x, out,
                         M, K,
                         M * K, K, 1,
                         M * K, K, 1,
                         start_token_position=pos, THETA=10000., BLOCK_SIZE_M=2, BLOCK_SIZE_K=1024)
        return out

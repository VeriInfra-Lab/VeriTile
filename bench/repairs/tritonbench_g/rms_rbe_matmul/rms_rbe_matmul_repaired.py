import math
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
    pid_m = pid // (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))
    pid_n = pid % (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))

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


@triton.jit
def rms_matmul_rbe(
        x_ptr, w_ptr, rms_w_ptr, out_ptr,
        M, N, K,
        stride_x_batch, stride_x_m, stride_x_k,
        stride_w_k, stride_w_n,
        stride_rms_w,
        stride_out_batch, stride_out_m, stride_out_n,
        start_token_position,
        USE_FP8: tl.constexpr,
        RBE_EPILOGUE: tl.constexpr,
        THETA: tl.constexpr,
        EPS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Prologue: RMS
    Epilogue: nothing or Rotary embeddings
    c = ROBE((rms(a) * rms_w) @ b)
    """
    M = tl.cast(M, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    stride_x_batch = tl.cast(stride_x_batch, tl.int64)
    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_k = tl.cast(stride_x_k, tl.int64)
    stride_w_k = tl.cast(stride_w_k, tl.int64)
    stride_w_n = tl.cast(stride_w_n, tl.int64)
    stride_rms_w = tl.cast(stride_rms_w, tl.int64)
    stride_out_batch = tl.cast(stride_out_batch, tl.int64)
    stride_out_m = tl.cast(stride_out_m, tl.int64)
    stride_out_n = tl.cast(stride_out_n, tl.int64)
    pid_batch = tl.program_id(axis=0).to(tl.int64)
    pid = tl.program_id(axis=1).to(tl.int64)
    pid_m = pid // (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))
    pid_n = pid % (N // BLOCK_SIZE_N + (N % BLOCK_SIZE_N != 0))

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_w_ptrs = rms_w_ptr + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)[None, :] * stride_rms_w
    x_sum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for step in range(0, (K // BLOCK_SIZE_K + (K % BLOCK_SIZE_K != 0))):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        x_sum += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)
        rms_w = tl.load(rms_w_ptrs, mask=offs_k[None, :] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            rms_w = rms_w.to(tl.float8e5, bitcast=True)
            rms_w = rms_w.to(tl.float16)
        x = x * rms_w
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - step * BLOCK_SIZE_K, other=0.0)
        if USE_FP8:
            w = w.to(tl.float8e5, bitcast=True)
            w = w.to(tl.float32)
            w = w.to(tl.float16)
        w = w.to(x.dtype)
        accumulator += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_x_k
        w_ptrs += BLOCK_SIZE_K * stride_w_k
        rms_w_ptrs += BLOCK_SIZE_K * stride_rms_w
    x_mean = tl.sum(x_sum, axis=1) / K + EPS
    x_norm = tl.math.rsqrt(x_mean)
    accumulator = accumulator * x_norm[:, None]

    # Rotate normalized FP32 pairs before the source's sole output cast.
    if RBE_EPILOGUE:
        real, imag = tl.split(tl.reshape(accumulator, (BLOCK_SIZE_M, BLOCK_SIZE_N // 2, 2)))
        pair_n = pid_n * BLOCK_SIZE_N + 2 * tl.arange(0, BLOCK_SIZE_N // 2).to(tl.int64)
        row_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        exponent = (pair_n % HEAD_DIM).to(tl.float32) / HEAD_DIM
        frequency = tl.extra.cuda.libdevice.pow(THETA, exponent)
        position = tl.cast(start_token_position, tl.int64) + row_m
        angle = position.to(tl.float32)[:, None] / frequency[None, :]
        cosine = tl.cos(angle)
        sine = tl.sin(angle)
        rotated_real = real * cosine - imag * sine
        rotated_imag = real * sine + imag * cosine
        accumulator = tl.reshape(tl.join(rotated_real, rotated_imag), (BLOCK_SIZE_M, BLOCK_SIZE_N))

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    out_ptrs = out_ptr + (
                pid_batch * stride_out_batch + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n)
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(out_ptrs, accumulator, mask=out_mask)


def _checked_tensor(tensor, name, dtypes):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(name + " must be a Tensor")
    if tensor.layout != torch.strided or tensor.dtype not in dtypes:
        raise TypeError(name + " has unsupported layout or dtype")
    if tensor.device.type != "cuda" or not tensor.is_contiguous() or tensor.is_neg():
        raise ValueError(name + " must be resolved, contiguous and on CUDA")
    if tensor.numel() and tensor.data_ptr() % tensor.element_size():
        raise ValueError(name + " has a misaligned data pointer")


def _checked_scalars(start_pos, n_heads, head_dim, batch, rows, rotary, eps, theta):
    if type(n_heads) is not int or type(head_dim) is not int or n_heads <= 0 or head_dim <= 0 or head_dim % 2:
        raise ValueError("head count must be positive and head width positive and even")
    if not isinstance(start_pos, int) or not -(1 << 63) <= start_pos < (1 << 63):
        raise ValueError("position must be a signed int64 Python integer")
    if rotary and batch and rows and start_pos + 16 * triton.cdiv(rows, 16) - 1 >= (1 << 63):
        raise ValueError("padded rotary positions must fit signed int64")
    for scalar in (eps, theta):
        try:
            valid = isinstance(scalar, (int, float)) and math.isfinite(scalar) and scalar > 0
        except (OverflowError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("epsilon and theta must be finite positive Python scalars")


def _overlaps(left, right):
    a, b = left.untyped_storage(), right.untyped_storage()
    return a.data_ptr() < b.data_ptr() + b.nbytes() and b.data_ptr() < a.data_ptr() + a.nbytes()


def rms_matmul_rbe_wrapper(x: torch.Tensor, weight: torch.Tensor, rms_w: torch.Tensor,
                           use_rbe: bool, start_pos: int, n_heads: int, head_dim: int):
    _checked_tensor(x, "x", (torch.float32,))
    _checked_tensor(weight, "weight", (torch.float16, torch.int8))
    _checked_tensor(rms_w, "rms_w", (torch.float32, torch.float16, torch.int8))
    if x.device != weight.device or x.device != rms_w.device:
        raise ValueError("inputs must share the input CUDA device")
    if type(use_rbe) is not bool:
        raise TypeError("use_rbe must be bool")
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError("input/projection ranks are incompatible")
    batch, M, K = x.shape
    N, K_W = weight.shape
    _checked_scalars(start_pos, n_heads, head_dim, batch, M, use_rbe, 1e-6, 10000.)
    if K == 0 or K_W != K or rms_w.shape != (K,) or N != n_heads * head_dim:
        raise ValueError("RMS/projection/head dimensions are incompatible")
    reads = (x, weight, rms_w)
    if any((a.dtype != b.dtype or a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr())
           and _overlaps(a, b) for i, a in enumerate(reads) for b in reads[i + 1:]):
        raise ValueError("overlapping input backings must share a base pointer and dtype")
    with torch.cuda.device(x.device):
        if rms_w.dtype != weight.dtype:
            rms_w = rms_w.to(weight.dtype)
        weight_t = weight.t()
        out = torch.empty((batch, M, N), dtype=weight.dtype, device=x.device)
        if batch and M:
            out_ptr = triton.reinterpret(out, tl.float8e5 if out.dtype == torch.int8 else tl.float16)
            rms_matmul_rbe[(batch, triton.cdiv(M, 16) * triton.cdiv(N, 64))](
                x_ptr=x, w_ptr=weight_t, rms_w_ptr=rms_w, out_ptr=out_ptr, M=M, N=N, K=K,
                stride_x_batch=M*K, stride_x_m=K, stride_x_k=1, stride_w_k=1, stride_w_n=K,
                stride_rms_w=1, stride_out_batch=M*N, stride_out_m=N, stride_out_n=1,
                start_token_position=start_pos, USE_FP8=weight.dtype == torch.int8,
                RBE_EPILOGUE=use_rbe, HEAD_DIM=head_dim, THETA=10000., EPS=1e-6,
                BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64, num_stages=4, num_warps=4)
        return out.view(batch, M, n_heads, head_dim)

"""Checked direct FP32 RMSNorm; original pow/rsqrt kernel is unchanged."""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_N = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)

    var = tl.sum(var, axis=0) / N_SIZE
    rstd = tl.math.rsqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        offs_n = block_n_start_idx + block_N
        x_ptr_mask = offs_n < N_SIZE
        rms_w = tl.load(rms_w_ptr + offs_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offs_m + offs_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + offs_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm_triton_wrapper(x, rms_w, eps=1e-6):
    if not isinstance(x, torch.Tensor) or not isinstance(rms_w, torch.Tensor):
        raise TypeError("x and rms_w must be tensors")
    if any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda
           for t in (x, rms_w)):
        raise TypeError("x and rms_w must be strided FP32 CUDA tensors")
    if x.device != rms_w.device:
        raise ValueError("x and rms_w must be on the same device")
    if x.ndim != 3 or x.shape[2] <= 0 or rms_w.shape != (x.shape[2],):
        raise ValueError("x must have rank three and a positive matching weight width")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (x, rms_w)):
        raise ValueError("inputs must be contiguous with no lazy flags")
    if type(eps) is not float or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive Python float")
    batch, M, K = x.shape
    padded = triton.cdiv(K, 1024) * 1024
    if x.numel() != 0:
        if batch >= 2**31 or M > 65535 or padded >= 2**31:
            raise ValueError("grid and final loop increment must fit the source index ABI")
        if any(v >= 2**31 for v in (*x.stride(), *rms_w.stride())):
            raise ValueError("source strides must fit signed 32-bit arguments")
        input_max = (batch - 1) * x.stride(0) + (M - 1) * x.stride(1) + (padded - 1) * x.stride(2)
        weight_max = (padded - 1) * rms_w.stride(0)
        if input_max >= 2**31 or weight_max >= 2**31:
            raise ValueError("all formed source offsets must fit signed 32-bit arithmetic")
        if x.data_ptr() % 4 != 0 or rms_w.data_ptr() % 4 != 0:
            raise ValueError("FP32 inputs must be 4-byte aligned")
        if x.data_ptr() + 4 * input_max >= 2**64 or rms_w.data_ptr() + 4 * weight_max >= 2**64:
            raise ValueError("formed input pointers must not wrap")
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    output_max = (batch - 1) * out.stride(0) + (M - 1) * out.stride(1) + (padded - 1) * out.stride(2)
    if any(v >= 2**31 for v in out.stride()) or output_max >= 2**31:
        raise ValueError("all formed output offsets must fit signed 32-bit arithmetic")
    if out.data_ptr() % 4 != 0 or out.data_ptr() + 4 * output_max >= 2**64:
        raise ValueError("formed output pointers must be aligned and must not wrap")
    with torch.cuda.device(x.device):
        rmsnorm_triton[(batch, M,)](x, rms_w, out,
                                  *x.stride(), *rms_w.stride(), *out.stride(),
                                  N_SIZE=K, eps=eps, BLOCK_N_SIZE=1024)
    return out

"""Checked direct FP32 RMSNorm; original numerical kernel body is unchanged."""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, out_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w,
                   stride_out_batch, stride_out_m, stride_out_k,
                   N_SIZE: tl.constexpr, eps: tl.constexpr, BLOCK_N_SIZE: tl.constexpr):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    # parallel at m dimension
    offset_m = pid_batch * stride_x_batch + pid_m * stride_x_m
    block_n_size = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)
    # parallel between blocks
    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)  # careful stride_x_k
        xf = x.to(tl.float32)
        var += xf*xf
    var = tl.sum(var, axis=0) / N_SIZE  # reduce between wrap
    std = tl.sqrt(var + eps)

    for block_n_strart_ptr in range(0, N_SIZE, BLOCK_N_SIZE):
        offset_n = block_n_strart_ptr + block_n_size
        x_ptr_mask = offset_n < N_SIZE

        rms_w_offset = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.)

        x_new = x / std
        out = x_new * rms_w_offset
        out_offset = pid_batch * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(out_ptr + out_offset, out, mask=x_ptr_mask)


def rmsnorm_wrapper(x, rms_weights, eps=1e-6):
    if not isinstance(x, torch.Tensor) or not isinstance(rms_weights, torch.Tensor):
        raise TypeError("x and rms_weights must be tensors")
    if any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda
           for t in (x, rms_weights)):
        raise TypeError("x and rms_weights must be strided FP32 CUDA tensors")
    if x.device != rms_weights.device:
        raise ValueError("x and rms_weights must be on the same device")
    if x.ndim != 3 or x.shape[2] <= 0 or rms_weights.shape != (x.shape[2],):
        raise ValueError("x must have rank three and a positive matching weight width")
    if any(not t.is_contiguous() or t.is_neg() or t.is_conj() for t in (x, rms_weights)):
        raise ValueError("inputs must be contiguous with no lazy flags")
    if type(eps) is not float or not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be a finite positive Python float")
    batch, M, K = x.shape
    padded = triton.cdiv(K, 4096) * 4096
    if x.numel() != 0:
        if batch >= 2**31 or M > 65535 or padded >= 2**31:
            raise ValueError("grid and final loop increment must fit the source index ABI")
        if any(v >= 2**31 for v in (*x.stride(), *rms_weights.stride())):
            raise ValueError("source strides must fit signed 32-bit arguments")
        input_max = (batch - 1) * x.stride(0) + (M - 1) * x.stride(1) + (padded - 1) * x.stride(2)
        weight_max = (padded - 1) * rms_weights.stride(0)
        if input_max >= 2**31 or weight_max >= 2**31:
            raise ValueError("all formed source offsets must fit signed 32-bit arithmetic")
        if x.data_ptr() % 4 != 0 or rms_weights.data_ptr() % 4 != 0:
            raise ValueError("FP32 inputs must be 4-byte aligned")
        if x.data_ptr() + 4 * input_max >= 2**64 or rms_weights.data_ptr() + 4 * weight_max >= 2**64:
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
        rmsnorm_triton[(batch, M,)](x, rms_weights, out,
                                  *x.stride(), *rms_weights.stride(), *out.stride(),
                                  N_SIZE=K, eps=eps, BLOCK_N_SIZE=4096, num_warps=16)
    return out

"""Checked FP32 RMSNorm preserving the source contiguous materialization."""
import torch
import triton
import math
import triton.language as tl

@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,  # pointer to the output
    X,  # pointer to the input
    W,  # pointer to the weights
    y_stride_r,
    y_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    Y += pid * y_stride_r
    X += pid * x_stride_r

    mask = tl.arange(0, BLOCK_SIZE) < N
    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols * x_stride_c, mask, other=0.0).to(tl.float32)

    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)

    w = tl.load(W + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = (x * rrms).to(Y.dtype.element_ty) * w
    tl.store(Y + cols * y_stride_c, y, mask=mask)


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
            raise TypeError("x and weight must be tensors")
        if any(t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda for t in (x, weight)):
            raise TypeError("x and weight must be strided FP32 CUDA tensors")
        if x.device != weight.device:
            raise ValueError("x and weight must be on the same device")
        if not isinstance(normalized_shape, (tuple, list)) or any(type(d) is not int or d < 0 for d in normalized_shape):
            raise TypeError("normalized_shape must contain nonnegative builtin integers")
        if len(normalized_shape) > x.ndim or tuple(x.shape[x.ndim-len(normalized_shape):]) != tuple(normalized_shape) or tuple(weight.shape) != tuple(normalized_shape) or math.prod(normalized_shape) <= 0:
            raise ValueError("normalized_shape must be a positive matching suffix and weight shape")
        if any(t.is_neg() or t.is_conj() for t in (x, weight)):
            raise ValueError("x and weight must have no lazy flags")
        if type(eps) is not float or not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a finite positive Python float")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        if M >= 2**31 or N >= 2**31:
            raise ValueError("rows and columns must fit signed 32-bit source arguments")
        BLOCK_SIZE = triton.next_power_of_2(N)
        if BLOCK_SIZE >= 2**31 or max(0, M-1) * N >= 2**31:
            raise ValueError("block lanes and row products must fit signed 32-bit indexing")
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)

        max_offset = max(0, M-1) * N + BLOCK_SIZE - 1
        if any(t.data_ptr() % 4 != 0 for t in (x, weight, y)):
            raise ValueError("copied FP32 inputs and output must be aligned")
        if x.data_ptr() + 4 * max_offset >= 2**64 or weight.data_ptr() + 4 * (BLOCK_SIZE-1) >= 2**64 or y.data_ptr() + 4 * max_offset >= 2**64:
            raise ValueError("all formed padded pointers must not wrap")
        with torch.cuda.device(x.device):
            rms_norm_kernel[M,](y, x, weight, N, 1, N, 1, N, eps, BLOCK_SIZE)
        return y


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)




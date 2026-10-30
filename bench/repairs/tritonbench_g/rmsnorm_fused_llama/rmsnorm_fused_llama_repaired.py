
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = x * rstd
        y = x_hat * w
        # Write output
        tl.store(Y + cols, y.to(tl.float16), mask=mask)

def rmsnorm_forward(x, weight, eps):
    """Contiguous FP32 RMSNorm, retaining the source's FP16 output round trip."""
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        raise TypeError("x and weight must be tensors")
    if x.layout != torch.strided or weight.layout != torch.strided:
        raise ValueError("x and weight must be strided tensors")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("x and weight must be float32")
    if x.ndim < 1 or weight.ndim != 1:
        raise ValueError("x needs a feature axis and weight must be one-dimensional")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if x.is_neg() or x.is_conj() or weight.is_neg() or weight.is_conj():
        raise ValueError("lazy tensor metadata is unsupported")
    if x.device != weight.device or not x.is_cuda:
        raise ValueError("x and weight must share a CUDA device")
    N = x.shape[-1]
    if N <= 0 or N > 16384 or weight.numel() != N:
        raise ValueError("feature size must be between 1 and 16384 and match weight")
    if x.numel() + 16383 >= 2**31:
        raise ValueError("padded element offsets must fit signed 32-bit indexing")
    if type(eps) is not float or not 0 < eps < float("inf"):
        raise ValueError("eps must be a finite positive Python float")
    if x.numel() and (x.data_ptr() % 16 or weight.data_ptr() % 16):
        raise ValueError("nonempty RMSNorm inputs require 16-byte aligned pointers")
    y = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.numel() == 0:
        return y
    x_arg = x.view(-1, N)
    M = x_arg.shape[0]
    with torch.cuda.device(x.device):
        _rms_norm_fwd_fused[(M,)](x_arg, y, weight, N, N, eps,
                                BLOCK_SIZE=16384, num_warps=8)
    return y

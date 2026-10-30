# Separate repair of original SHA256:
# 3b190a21ae5ef1c37f8d24e16c1757950215749d28a463f3c1989657fa07a017
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel_repaired(
    X, Y, N, eps, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    X += row * N
    Y += row * N
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    y = x * rstd
    tl.store(Y + cols, y, mask=mask)


def _l2_norm_fwd_repaired(x, eps=1e-6):
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a tensor")
    if x.layout != torch.strided:
        raise TypeError("x must have strided layout")
    if x.dtype != torch.float32:
        raise TypeError("x must have dtype torch.float32")
    if not x.is_cuda:
        raise ValueError("x must be a CUDA tensor")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    N = x.shape[-1]
    if N == 0 or N > 16384:
        raise ValueError("last dimension must be in [1, 16384]")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if x.is_neg():
        raise ValueError("x must not have an unresolved negative view bit")
    if x.numel() > 0 and x.data_ptr() % 16 != 0:
        raise ValueError("nonempty x must have a 16-byte-aligned data pointer")
    x_shape_og = x.shape
    with torch.cuda.device(x.device):
        x = x.view(-1, N)
        y = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        M = x.shape[0]
        BLOCK_N = triton.next_power_of_2(N)
        if M:
            _l2_norm_fwd_1pass_kernel_repaired[(M,)](
                x, y, N, eps, BLOCK_N, num_warps=4, num_ctas=1,
            )
    return y.view(x_shape_og)

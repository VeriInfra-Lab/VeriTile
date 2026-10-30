# Separate repair of original SHA256:
# 386a7d3abb4c3960d6af1ff90cb29c0733c0ac6e9ed51319bd22309568947c12
import torch
import triton
import triton.language as tl


@triton.jit
def _l2_norm_fwd_1pass_kernel_repaired(X, Y, N, eps, BLOCK_N: tl.constexpr):
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


@triton.jit
def _l2_norm_bwd_kernel_repaired(X, DY, DX, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    X += row * N
    DX += row * N
    DY += row * N
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x)
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var + eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)


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


def _l2_norm_bwd_repaired(x, dy, eps=1e-5):
    if not isinstance(x, torch.Tensor) or not isinstance(dy, torch.Tensor):
        raise TypeError("x and dy must be tensors")
    if x.layout != torch.strided or dy.layout != torch.strided:
        raise TypeError("x and dy must have strided layout")
    if x.dtype != torch.float32 or dy.dtype != torch.float32:
        raise TypeError("x and dy must have dtype torch.float32")
    if not x.is_cuda or not dy.is_cuda:
        raise ValueError("x and dy must be CUDA tensors")
    if x.device != dy.device:
        raise ValueError("x and dy must be on the same device")
    if dy.ndim == 0:
        raise ValueError("dy must have at least one dimension")
    N = dy.shape[-1]
    if N == 0 or N > 16384:
        raise ValueError("dy last dimension must be in [1, 16384]")
    if x.numel() != dy.numel():
        raise ValueError("x and dy must contain the same number of elements")
    if not x.is_contiguous() or not dy.is_contiguous():
        raise ValueError("x and dy must be contiguous")
    if x.is_neg() or dy.is_neg():
        raise ValueError("x and dy must not have unresolved negative view bits")
    if x.numel() > 0 and (x.data_ptr() % 16 != 0 or dy.data_ptr() % 16 != 0):
        raise ValueError("nonempty inputs must have 16-byte-aligned data pointers")
    x_shape_og = x.shape
    with torch.cuda.device(x.device):
        x = x.view(-1, N)
        dy = dy.view(-1, N)
        dx = torch.empty(x.shape, dtype=torch.float32, device=x.device)
        M = x.shape[0]
        BLOCK_N = triton.next_power_of_2(N)
        if M:
            _l2_norm_bwd_kernel_repaired[(M,)](
                x, dy, dx, N, eps, BLOCK_N, num_warps=4, num_ctas=1,
            )
    return dx.view(x_shape_og)

"""Checked direct-view repair; accepted kernel and native FP32 graph retained."""
import torch
import triton
import triton.language as tl

# Keep the native lane layout independent of view alignment and scalar values.
@triton.jit(
    do_not_specialize=["X", "DY", "DX", "stride_x_row", "stride_x_col",
                       "stride_dy_row", "stride_dy_col", "stride_dx_row",
                       "stride_dx_col", "N", "eps"],
    do_not_specialize_on_alignment=["X", "DY", "DX", "stride_x_row", "stride_x_col",
                                    "stride_dy_row", "stride_dy_col", "stride_dx_row",
                                    "stride_dx_col", "N", "eps"],
)
def _l2_norm_bwd_kernel_repaired(
    X,  # pointer to the input
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row: tl.int64,
    stride_x_col: tl.int64,
    stride_dy_row: tl.int64,
    stride_dy_col: tl.int64,
    stride_dx_row: tl.int64,
    stride_dx_col: tl.int64,
    N: tl.int64,  # number of logical columns
    eps: tl.float32,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    X += row * stride_x_row
    DX += row * stride_dx_row
    DY += row * stride_dy_row
    cols = tl.arange(0, BLOCK_N).to(tl.int64)
    x = tl.load(X + cols * stride_x_col, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x) 
    rstd = 1 / tl.sqrt(var + eps)
    mask = cols < N
    dy = tl.load(DY + cols * stride_dy_col, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols * stride_dx_col, dx, mask=mask)

def _l2_norm_bwd(
    x, dy, eps=1e-5,
):
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
    if x.numel() % N != 0:
        raise ValueError("x must divide into complete DY-width rows")
    if dy.numel() < x.numel():
        raise ValueError("dy must contain the logical prefix used by x")
    if not x.is_contiguous() or not dy.is_contiguous():
        raise ValueError("x and dy must be contiguous")
    if x.is_neg() or dy.is_neg():
        raise ValueError("x and dy must not have unresolved negative view bits")
    if x.numel() > 0 and (x.data_ptr() % 4 != 0 or dy.data_ptr() % 4 != 0):
        raise ValueError("nonempty read pointers must be aligned for FP32 scalar accesses")
    x_shape_og = x.shape
    x = x.view(-1, N)
    dy = dy.view(-1, N)
    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    if dy.numel() < x.numel():
        raise ValueError("l2_norm_bwd requires at least x.numel() logical DY elements")
    if x.device != dy.device:
        raise ValueError("l2_norm_bwd requires the same device for X and DY")
    with torch.cuda.device(x.device.index):
        x_values = x
        dy_values = dy
        _l2_norm_bwd_kernel_repaired[(M,)](
            x_values,
            dy_values,
            dx,
            x_values.stride(0),
            x_values.stride(1),
            dy_values.stride(0),
            dy_values.stride(1),
            dx.stride(0),
            dx.stride(1),
            N,
            eps,
            BLOCK_N,
        )
    return dx.view(x_shape_og)


import math
import triton
import triton.language as tl
import torch
import torch.nn as nn


next_power_of_2 = triton.next_power_of_2
MAX_FUSED_SIZE : int = 65536

def calculate_settings(n : int) -> (int, int,):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def _rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets * W_row_stride, mask=mask, other=0)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask=mask)

@triton.heuristics({"GEMMA": lambda args: args["GEMMA"],})
@triton.jit
def _rms_layernorm_backward(
    dY, dY_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    dW, dW_row_stride,
    n_cols, eps,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X  += row_idx *  X_row_stride
    r  += row_idx *  r_row_stride

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row  = tl.load(X  + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row  = tl.load(W  + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA: dY_W = dY_row * (W_row + 1.0)
    else:     dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    output = inv_var/n_cols * (n_cols*dY_W - normed*rowsum_dY_normed)
    tl.store(dY + col_offsets, output, mask=mask)

@triton.jit
def _gemma_rms_layernorm_forward(
    Y, Y_row_stride,
    X, X_row_stride,
    W, W_row_stride,
    r, r_row_stride,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask=mask)

@triton.jit
def _rms_layernorm_weight_backward(
    dW, dY, X, r,
    n_rows, n_cols,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # One owner per weight lane, with a deterministic increasing-row fold.
    col = tl.program_id(0).to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col < n_cols
    acc = tl.full((BLOCK_SIZE,), 0, tl.float32)
    for row in range(tl.cast(n_rows, tl.int64)):
        dy = tl.load(dY + row * n_cols + col, mask=mask, other=0).to(tl.float32)
        x = tl.load(X + row * n_cols + col, mask=mask, other=0).to(tl.float32)
        inv = tl.load(r + row).to(tl.float32)
        normed = x * inv
        if not GEMMA:
            normed = normed.to(dW.dtype.element_ty).to(tl.float32)
        acc = acc + dy * normed
    tl.store(dW + col, acc, mask=mask)


def _snapshot(tensor, shape):
    # The input-gradient kernel writes this private buffer in place.
    # Keep caller upstream storage read-only while dW reads its original values.
    return tensor.view(shape).clone(memory_format=torch.contiguous_format)


def _check_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("RMSNorm requires tensors")
    if tensor.layout != torch.strided:
        raise ValueError("RMSNorm requires strided tensors")
    if tensor.dtype != torch.float32:
        raise TypeError("RMSNorm requires FP32 tensors")
    if not tensor.is_cuda:
        raise ValueError("RMSNorm requires CUDA tensors")
    if torch.cuda.current_device() != tensor.device.index:
        raise ValueError("RMSNorm requires tensors on the current CUDA device")
    if not tensor.is_contiguous():
        raise ValueError("RMSNorm requires contiguous tensors")
    if tensor.is_neg():
        raise ValueError("RMSNorm does not accept unresolved negative views")


def _check_forward(X, W, eps, gemma):
    _check_tensor(X)
    _check_tensor(W)
    if X.device != W.device:
        raise ValueError("RMSNorm input and weight must share a device")
    if X.ndim == 0 or X.shape[-1] == 0:
        raise ValueError("RMSNorm requires a nonempty final dimension")
    if X.shape[-1] > MAX_FUSED_SIZE:
        raise ValueError("RMSNorm feature count exceeds 65536")
    if W.numel() != X.shape[-1]:
        raise ValueError("RMSNorm weight count must equal the feature count")
    if X.numel() // X.shape[-1] > 2147483647:
        raise ValueError("RMSNorm row count exceeds the CUDA x-grid limit")
    if X.untyped_storage()._cdata == W.untyped_storage()._cdata:
        raise ValueError("RMSNorm input and weight require distinct backing storage")
    if (X.numel() and X.data_ptr() % 16) or W.data_ptr() % 16:
        raise ValueError("RMSNorm inputs require 16-byte aligned pointers")
    if type(eps) is not float:
        raise TypeError("RMSNorm epsilon must be a Python float")
    if not math.isfinite(eps):
        raise ValueError("RMSNorm epsilon must be finite")
    if type(gemma) is not bool:
        raise TypeError("RMSNorm gemma must be a bool")


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, eps, gemma=False):
        _check_forward(X, W, eps, gemma)
        shape = tuple(X.shape)
        if not shape or shape[-1] == 0:
            raise ValueError("RMSNorm requires a nonempty final dimension")
        dim = shape[-1]
        n_rows = X.numel() // dim
        BLOCK_SIZE, num_warps = calculate_settings(dim)
        if X.device != W.device:
            raise ValueError("RMSNorm input and weight must share a device")
        weight_shape = tuple(W.shape)
        X = X.view((n_rows, dim))
        W = W.view((dim,))
        Y = torch.empty_like(X)
        r = torch.empty(n_rows, dtype=torch.float32, device=X.device)

        if n_rows:
            fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
            fx[(n_rows,)](
                Y, dim,
                X, dim,
                W, 1,
                r, 1,
                dim, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.GEMMA = gemma
        ctx.input_shape = shape
        ctx.weight_shape = weight_shape
        ctx.save_for_backward(X, W, r)
        return Y.view(shape)

    @staticmethod
    def backward(ctx, dY):
        X, W, r = ctx.saved_tensors
        _check_tensor(dY)
        n_rows, n_cols = X.shape
        if dY.device != X.device:
            raise ValueError("RMSNorm gradient and saved input must share a device")
        if dY.numel() != n_rows * n_cols:
            raise ValueError("RMSNorm gradient count must equal the saved input count")
        dX = _snapshot(dY, (n_rows, n_cols))
        dW = torch.empty_like(W)
        if n_rows:
            _rms_layernorm_weight_backward[(triton.cdiv(n_cols, 256),)](
                dW, dX, X, r, n_rows, n_cols,
                GEMMA=ctx.GEMMA,
                BLOCK_SIZE=256,
                enable_fp_fusion=False,
            )
            _rms_layernorm_backward[(n_rows,)](
                dX, n_cols,
                X, n_cols,
                W, 1,
                r, 1,
                dW, 1,
                n_cols, ctx.eps,
                GEMMA=ctx.GEMMA,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )
        else:
            dW.zero_()
        return dX.view(ctx.input_shape), dW.view(ctx.weight_shape), None, None

def fast_rms_layernorm(layernorm, X, gemma=False):
    W = layernorm.weight
    eps = layernorm.variance_epsilon if \
        hasattr(layernorm, "variance_epsilon") \
        else layernorm.eps
    out = Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    return out


# Define a simple layernorm module for testing
class SimpleLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(SimpleLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape).cuda())
        self.eps = eps



##################################################################################################################################################


# Test function for the fast_rms_layernorm with backward pass
def test_fast_rms_layernorm_with_backward():
    # Create a random input tensor with gradient tracking
    X = torch.randn(2, 4, 8, device='cuda', dtype=torch.float32, requires_grad=True)

    # Create a layernorm instance
    layernorm = SimpleLayerNorm(normalized_shape=X.shape[-1])

    results = {}

    # Test without GEMMA
    output = fast_rms_layernorm(layernorm, X, gemma=False)
    loss = output.mean()  # Compute a dummy loss

    # Perform backward propagation
    loss.backward()
    results['test_case_1'] = X.grad.clone()

    # Reset gradients for the next test
    X.grad.zero_()

    # Test with GEMMA
    output_gemma = fast_rms_layernorm(layernorm, X, gemma=True)
    loss_gemma = output_gemma.mean()  # Compute a dummy loss

    # Perform backward propagation
    loss_gemma.backward()
    results['test_case_2'] = X.grad.clone()

    return results

# Run the test
result_gold = test_fast_rms_layernorm_with_backward()

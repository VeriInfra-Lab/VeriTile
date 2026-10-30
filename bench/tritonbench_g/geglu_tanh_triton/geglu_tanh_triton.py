"""Direct FP32 flat-grid GEGLU repair with original scalar expressions."""
import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

@triton.jit
def _geglu_tanh_forward_kernel_repaired(
    a, b, c, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)


    col_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_elements
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)

@triton.jit
def _geglu_tanh_backward_kernel_repaired(
    dc, a, b, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)


    col_offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = col_offsets < n_elements

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)

    db_row = dc_row * geglu_a

    # Gradient w.r.t. a can be computed with:
    # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
    # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * a_row
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    )
    da_row = dc_row * b_row * (term1 + term2)

    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)

def _check_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a tensor")
    if tensor.layout != torch.strided or tensor.dtype != torch.float32:
        raise TypeError("expected strided FP32 storage")
    if not tensor.is_cuda or not tensor.is_contiguous() or tensor.is_neg():
        raise ValueError("expected contiguous CUDA storage without a negative view")


def _check_call(tensors, shaped):
    for tensor in tensors:
        _check_tensor(tensor)
    if shaped.ndim == 0 or shaped.shape[-1] == 0:
        raise ValueError("expected a positive final dimension")
    n = shaped.numel()
    cols = shaped.shape[-1]
    rows = n // cols
    if any(t.device != shaped.device or t.numel() != n for t in tensors):
        raise ValueError("payloads must have matching device and element count")
    if triton.cdiv(n, 128) >= 2**31 or n + 127 >= 2**63:
        raise ValueError("launch index bound exceeded")
    if n > 0:
        if any(t.data_ptr() % 4 for t in tensors):
            raise ValueError("active FP32 pointers must be aligned")
        for i, left in enumerate(tensors):
            for right in tensors[i+1:]:
                if not (left.data_ptr() + 4*n <= right.data_ptr() or
                        right.data_ptr() + 4*n <= left.data_ptr()):
                    raise ValueError("payload spans must be disjoint")
    return rows, cols


def geglu_forward_repaired(a, b):
    _, n_cols = _check_call([a, b], a)
    if a.shape != b.shape:
        raise ValueError("forward shapes must match")
    ori_shape = a.shape
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    count = a.numel()
    with torch.cuda.device(a.device):
        c = torch.empty_like(a)
        if count:
            _geglu_tanh_forward_kernel_repaired[(triton.cdiv(count, 128),)](
                a, b, c, n_elements=count, BLOCK_SIZE=128, num_warps=4, num_ctas=1)
    return a, b, c.view(*ori_shape)


def geglu_backward_repaired(a, b, dc):
    _, n_cols = _check_call([a, b, dc], dc)
    ori_shape = dc.shape
    dc = dc.view(-1, n_cols)
    count = dc.numel()
    with torch.cuda.device(dc.device):
        if count:
            _geglu_tanh_backward_kernel_repaired[(triton.cdiv(count, 128),)](
                dc, a, b, n_elements=count, BLOCK_SIZE=128, num_warps=4, num_ctas=1)
    return a.view(*ori_shape), b.view(*ori_shape)

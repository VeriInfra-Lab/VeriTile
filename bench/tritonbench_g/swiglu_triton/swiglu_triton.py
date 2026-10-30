"""Direct-storage FP32 repair; original kernels and precision are retained."""

import torch
import triton
import triton.language as tl
import functools

def is_hip() -> bool:
    return torch.version.hip is not None

def calculate_settings(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)

@triton.jit
def _swiglu_backward_kernel(
    dc_ptr, a_ptr, b_ptr, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    # locate start index
    dc_ptr += program_id * stride
    a_ptr += program_id * stride
    b_ptr += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc_ptr + col_offsets, mask=mask, other=0)
    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sig_a = tl.sigmoid(a_row)
    silu_a = a_row * sig_a
    db_row = dc_row * silu_a
    da_row = dc_row * (silu_a * (1 - sig_a) + sig_a) * b_row

    tl.store(a_ptr + col_offsets, da_row, mask=mask)
    tl.store(b_ptr + col_offsets, db_row, mask=mask)

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
    if rows >= 2**31 or n + 65536 >= 2**63:
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


def swiglu_forward(a, b):
    n_rows, n_cols = _check_call([a, b], a)
    if a.shape != b.shape:
        raise ValueError("forward shapes must match")
    ori_shape = a.shape
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    with torch.cuda.device(a.device):
        c = torch.empty_like(a)
        if n_rows > 0:
            _swiglu_forward_kernel[(n_rows,)](
                a, b, c, n_cols, n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return a, b, c.view(*ori_shape)


def swiglu_backward(a, b, dc):
    n_rows, n_cols = _check_call([a, b, dc], dc)
    ori_shape = dc.shape
    dc = dc.view(-1, n_cols)
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    with torch.cuda.device(dc.device):
        if n_rows > 0:
            _swiglu_backward_kernel[(n_rows,)](
                dc, a, b, n_cols, n_cols=n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return a.view(*ori_shape), b.view(*ori_shape)

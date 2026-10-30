"""Separate FP32 repair; contiguous rank-two inputs, empty shapes and widened addresses.
The four loops and default fusion policy are retained; local -inf guards avoid
spurious NaNs in rows that contain finite values.
"""
import triton
from triton import language as tl
import torch

@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b

@triton.jit
def softmax_kernel_online_v2_repaired(
    output_ptr,
    input_ptr,
    M: tl.int64,
    N: tl.int64,
    TILE_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    m = tl.full((TILE_N,), value=-float("inf"), dtype=output_ptr.dtype.element_ty)
    z = tl.full((TILE_N,), value=0, dtype=output_ptr.dtype.element_ty)
    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
        new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
        new_z = old_mass + new_mass
        m = new_m
        z = new_z
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        new_m = tl.maximum(m, inp)
        old_mass = tl.where(new_m == -float("inf"), z, tl.exp(m - new_m) * z)
        new_mass = tl.where(inp == -float("inf"), 0.0, tl.exp(inp - new_m))
        new_z = old_mass + new_mass
        m = new_m
        z = new_z
    final_m = tl.max(m, 0)
    z = tl.sum(tl.exp(m - final_m) * z)
    m = final_m

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, prev_multiple, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out)
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = pid_m * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)

def _check_input(x):
    if not isinstance(x, torch.Tensor):
        raise TypeError("expected a Tensor")
    if x.layout != torch.strided or x.dtype != torch.float32 or not x.is_cuda:
        raise ValueError("expected a strided FP32 CUDA tensor")
    if x.ndim != 2:
        raise ValueError("expected a matrix")
    if not x.is_contiguous() or x.is_neg():
        raise ValueError("expected contiguous storage without negative view bits")
    if x.numel() >= 2**31:
        raise ValueError("numel must be below 2**31")
    if x.numel() and x.data_ptr() % 4:
        raise ValueError("FP32 pointers must be 4-byte aligned")


def softmax_repaired(x):
    _check_input(x)
    M, N = x.shape
    out = torch.empty_like(x)
    if M == 0 or N == 0:
        return out
    TILE_N = min(4096, 1 << (N.bit_length() - 1))
    grid = (M, 1, 1)
    with torch.cuda.device(x.device):
        softmax_kernel_online_v2_repaired[grid](
            out, x, M, N, TILE_N, enable_fp_fusion=True,
            num_warps=4, num_stages=3,
        )
    return out

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
def softmax_kernel_online_v2_index_commuted(
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
        offset = N * pid_m + n_offsets
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
        offset = N * pid_m + n_offsets
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
        offset = N * pid_m + n_offsets
        input_ptrs = input_ptr + offset
        inp = tl.load(input_ptrs).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out)
    for start_n in range(prev_multiple, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N).to(tl.int64)
        offset = N * pid_m + n_offsets
        input_ptrs = input_ptr + offset
        mask = n_offsets < N
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(output_ptr.dtype.element_ty)
        e = tl.exp(inp - m)
        out = e / z
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)


"""Authored Python counterpart of the existing Lean unused-count fold.

The selected original/helper are unchanged. Only the unused integer tile-count
expression is folded; this is not a recovered historical source or performance claim.
"""
import triton
import triton.language as tl
from .relu_strided_buffer_repaired import relu_forward

@triton.jit
def relu_forward_kernel_rank_1_folded_count(in0_ptr: tl.tensor, out0_ptr: tl.tensor, in0_stride0: int, in0_stride_order0: tl.constexpr, out0_stride0: int, out0_stride_order0: tl.constexpr, s0: int, num_tasks: int, tiles_per_cta: int, tile_size0: tl.constexpr, one_tile_per_cta: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles0 = (s0 + tile_size0 - 1) // tile_size0
    if one_tile_per_cta:
        tile_id = pid
        tile_id0 = tile_id
        offset0 = tile_id0 * tile_size0
        in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty)
        out0 = relu_forward(in0)
        out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
    else:
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            tile_id0 = tile_id
            offset0 = tile_id0 * tile_size0
            in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty)
            out0 = relu_forward(in0)
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))

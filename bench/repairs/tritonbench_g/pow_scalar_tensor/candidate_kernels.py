"""Existing integer tile-offset commutation, with both original branches retained."""
import math
from typing import Union
import torch
import triton
from triton import language as tl

try:
    from triton.language.extra.cuda.libdevice import pow as _pow
except ImportError:
    try:
        from triton.language.math import pow as _pow
    except ImportError:
        from triton.language.libdevice import pow as _pow

@triton.jit
def pow_func_scalar_tensor(x, exponent):
    return _pow(x.to(tl.float32), exponent)

@triton.jit(do_not_specialize=['val0'])
def pow_func_scalar_tensor_kernel_rank_1_index_commuted(
    val0,
    in0_ptr: tl.tensor, # of tl.pointer_type
    out0_ptr: tl.tensor, # of tl.pointer_type
    in0_stride0: int, # strides for in0
    in0_stride_order0: tl.constexpr, # stride order for in0
    out0_stride0: int, # strides for out0
    out0_stride_order0: tl.constexpr, # stride order for out0
    s0: int, # task_space
    num_tasks: int,
    tiles_per_cta: int,
    tile_size0: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv(s0, tile_size0)
    if one_tile_per_cta: # monolitic kernel style
        tile_id = pid
        # pid multi index recontruction: we use c ordering, right axes changes fastest
        tile_id0 = tile_id

        # tile offsets
        offset0 = tile_size0 * tile_id0
        # loads
        in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

        # compute
        out0 = pow_func_scalar_tensor(val0, in0)

        # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
        out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))
    else: # grid-stride-loop style kernel
        num_ctas = tl.num_programs(0)
        for j in range(0, tiles_per_cta):
            tile_id = pid + j * num_ctas
            # pid multi index recontruction: we use c ordering, right axes changes fastest
            tile_id0 = tile_id

            # tile offsets
            offset0 = tile_size0 * tile_id0
            # loads
            in0_bptr = tl.make_block_ptr(in0_ptr, (s0,), (in0_stride0,), (offset0,), (tile_size0,), order=(in0_stride_order0,))
            in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0,)).to(in0_ptr.type.element_ty) # workaround the bug on bool, we should use the original pointer's dtype(instead of block pointer's)

            # compute
            out0 = pow_func_scalar_tensor(val0, in0)

            # stores, note that store to block pointer does not automatically cast the value to the pointer's dtype
            out0_bptr = tl.make_block_ptr(out0_ptr, (s0,), (out0_stride0,), (offset0,), (tile_size0,), order=(out0_stride_order0,))
            tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0,))

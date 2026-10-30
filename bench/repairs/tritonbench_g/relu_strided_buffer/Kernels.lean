import VeriTile.Triton.DSL
import VeriTile.Triton.DSL.CallSignature

/-! Complete unchanged original ReLU helper/writer and the existing unused-count
candidate. Invocation storage, signed pointer origins/strides and constexpr
settings are outside the pasted bodies. -/
namespace VeriTile.Bench.Repairs.ReluStridedBuffer
open Lean VeriTile.Triton

structure View where
  region : RegionName
  base : Int
  stride : Int
structure Parameters where
  input : View
  output : View
  n : Nat
  B : Nat
  tiles : Nat
  oneTile : Bool

def reluFunction : SourceFunction :=
  triton_function relu_forward(x) in triton {
    return tl.where(x > 0, x, 0)
  }

def callSignature : CallSignature where
  pointers := [("in0_ptr",.fp32),("out0_ptr",.fp32)]
  scalars := [("in0_stride0",.integer),("out0_stride0",.integer),("s0",.integer),
    ("num_tasks",.integer),("tiles_per_cta",.integer)]
  constexpr := ["in0_stride_order0","out0_stride_order0","tile_size0","one_tile_per_cta"]
  typedFP32Storage := Bool.true
  functions := [reluFunction]

def originalBody (c : Parameters) : ComputeKernel :=
  let in0_ptr := c.input.region
  let out0_ptr := c.output.region
  let in0_stride_order0 : Nat := 0
  let out0_stride_order0 : Nat := 0
  let tile_size0 := c.B
  let one_tile_per_cta := c.oneTile
  with_triton_call callSignature call_args [
    in0_stride0 := c.input.stride, out0_stride0 := c.output.stride, s0 := c.n, num_tasks := c.n, tiles_per_cta := c.tiles]
    pointer_offsets [in0_ptr := c.input.base, out0_ptr := c.output.base] in triton {
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv(s0, tile_size0)
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
  }

def foldedBody (c : Parameters) : ComputeKernel :=
  let in0_ptr := c.input.region
  let out0_ptr := c.output.region
  let in0_stride_order0 : Nat := 0
  let out0_stride_order0 : Nat := 0
  let tile_size0 := c.B
  let one_tile_per_cta := c.oneTile
  with_triton_call callSignature call_args [
    in0_stride0 := c.input.stride, out0_stride0 := c.output.stride, s0 := c.n, num_tasks := c.n, tiles_per_cta := c.tiles]
    pointer_offsets [in0_ptr := c.input.base, out0_ptr := c.output.base] in triton {
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
  }

def oneTileKernel (c : Parameters) : ComputeKernel :=
  originalBody {c with oneTile := Bool.true}
def gridStrideKernel (c : Parameters) : ComputeKernel :=
  originalBody {c with oneTile := Bool.false}
def oneTileFolded (c : Parameters) : ComputeKernel :=
  foldedBody {c with oneTile := Bool.true}
def gridStrideFolded (c : Parameters) : ComputeKernel :=
  foldedBody {c with oneTile := Bool.false}

def kernel (c : Parameters) : ComputeKernel := originalBody c
def foldedKernel (c : Parameters) : ComputeKernel := foldedBody c

end VeriTile.Bench.Repairs.ReluStridedBuffer

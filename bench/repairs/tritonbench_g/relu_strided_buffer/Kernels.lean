import VeriTile.Triton.DSL

/-! Readable FP32 source for the two original constexpr branches. Rank-one
block pointers are lowered to their signed cell formula and logical mask.
The complete signed sum is formed before conversion to a memory cell index;
negative strides and arbitrary bounded storage offsets remain supported.
`DTYPE` is fixed FP32, including the pointer-element casts and zero conversion.
The candidate folds only the unused integer tile count. -/
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

def oneTileKernel (c : Parameters) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let input_region := c.input.region
  let output_region := c.output.region
  triton {
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv($(c.n), $(c.B))
    tile_id = pid
    tile_id0 = tile_id
    offset0 = tile_id0 * $(c.B)
    in0_bptr = $((c.input.base : Int)) +
      (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.input.stride : Int))
    in0 = (tl.load(input_region + in0_bptr,
      mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(DTYPE)
    out0 = tl.where(in0 > tl.cast(0.0, DTYPE), in0, tl.cast(0.0, DTYPE))
    out0_bptr = $((c.output.base : Int)) +
      (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
    tl.store(output_region + out0_bptr, (out0).to(DTYPE),
      mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
  }

def gridStrideKernel (c : Parameters) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let input_region := c.input.region
  let output_region := c.output.region
  triton {
    pid = tl.program_id(0)
    num_tiles0 = tl.cdiv($(c.n), $(c.B))
    num_ctas = tl.num_programs(0)
    for j in range($(0), $(c.tiles)) {
      tile_id = pid + j * num_ctas
      tile_id0 = tile_id
      offset0 = tile_id0 * $(c.B)
      in0_bptr = $((c.input.base : Int)) +
        (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.input.stride : Int))
      in0 = (tl.load(input_region + in0_bptr,
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(DTYPE)
      out0 = tl.where(in0 > tl.cast(0.0, DTYPE), in0, tl.cast(0.0, DTYPE))
      out0_bptr = $((c.output.base : Int)) +
        (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
      tl.store(output_region + out0_bptr, (out0).to(DTYPE),
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
    }
  }

def oneTileFolded (c : Parameters) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let input_region := c.input.region
  let output_region := c.output.region
  triton {
    pid = tl.program_id(0)
    num_tiles0 = $((c.n + c.B - 1) / c.B)
    tile_id = pid
    tile_id0 = tile_id
    offset0 = tile_id0 * $(c.B)
    in0_bptr = $((c.input.base : Int)) +
      (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.input.stride : Int))
    in0 = (tl.load(input_region + in0_bptr,
      mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(DTYPE)
    out0 = tl.where(in0 > tl.cast(0.0, DTYPE), in0, tl.cast(0.0, DTYPE))
    out0_bptr = $((c.output.base : Int)) +
      (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
    tl.store(output_region + out0_bptr, (out0).to(DTYPE),
      mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
  }

def gridStrideFolded (c : Parameters) : ComputeKernel :=
  let dtype : FloatDType := .fp32
  let input_region := c.input.region
  let output_region := c.output.region
  triton {
    pid = tl.program_id(0)
    num_tiles0 = $((c.n + c.B - 1) / c.B)
    num_ctas = tl.num_programs(0)
    for j in range($(0), $(c.tiles)) {
      tile_id = pid + j * num_ctas
      tile_id0 = tile_id
      offset0 = tile_id0 * $(c.B)
      in0_bptr = $((c.input.base : Int)) +
        (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.input.stride : Int))
      in0 = (tl.load(input_region + in0_bptr,
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(DTYPE)
      out0 = tl.where(in0 > tl.cast(0.0, DTYPE), in0, tl.cast(0.0, DTYPE))
      out0_bptr = $((c.output.base : Int)) +
        (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
      tl.store(output_region + out0_bptr, (out0).to(DTYPE),
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
    }
  }

def kernel (c : Parameters) : ComputeKernel :=
  if c.oneTile then oneTileKernel c else gridStrideKernel c

def foldedKernel (c : Parameters) : ComputeKernel :=
  if c.oneTile then oneTileFolded c else gridStrideFolded c

end VeriTile.Bench.Repairs.ReluStridedBuffer

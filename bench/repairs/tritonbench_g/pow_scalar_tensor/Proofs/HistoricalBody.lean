import VeriTile.Triton.DSL
/-! Historical flattened source retained only for migration checks. -/

/-! Shared readable scalar-base FP32 power source. Signed block pointers are
lowered to their complete signed cell formula and logical boundary mask.
The original and candidate differ only by commuting the integer tile-offset
product; all FP32 loads, scalar conversion, libdevice pow and stores remain.
`base` is the runtime scalar parameter, represented by the DSL scalar argument
slot; FP word semantics binds that slot to the launch ABI's FP32 word. -/
namespace VeriTile.Bench.Repairs.PowScalarTensor.Checked.HistoricalBody
open VeriTile.Triton

structure View where
  region : RegionName
  base : Int
  stride : Int
structure Parameters where
  input : View
  output : View
  n : Nat
  base : ℝ
  B : Nat
  tiles : Nat
  oneTile : Bool

def oneTileKernel : Parameters → ComputeKernel :=
  fun (c : Parameters) =>
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
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(tl.float32)
      out0 = tl.extra.cuda.libdevice.pow($((c.base : ℝ)).to(tl.float32), in0)
      out0_bptr = $((c.output.base : Int)) +
        (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
      tl.store(output_region + out0_bptr, (out0).to(DTYPE),
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
    }

def gridStrideKernel : Parameters → ComputeKernel :=
  fun (c : Parameters) =>
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
          mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(tl.float32)
        out0 = tl.extra.cuda.libdevice.pow($((c.base : ℝ)).to(tl.float32), in0)
        out0_bptr = $((c.output.base : Int)) +
          (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
        tl.store(output_region + out0_bptr, (out0).to(DTYPE),
          mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
      }
    }

def oneTileCommuted : Parameters → ComputeKernel :=
  fun (c : Parameters) =>
    let dtype : FloatDType := .fp32
    let input_region := c.input.region
    let output_region := c.output.region
    triton {
      pid = tl.program_id(0)
      num_tiles0 = tl.cdiv($(c.n), $(c.B))
      tile_id = pid
      tile_id0 = tile_id
      offset0 = $(c.B) * tile_id0
      in0_bptr = $((c.input.base : Int)) +
        (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.input.stride : Int))
      in0 = (tl.load(input_region + in0_bptr,
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(tl.float32)
      out0 = tl.extra.cuda.libdevice.pow($((c.base : ℝ)).to(tl.float32), in0)
      out0_bptr = $((c.output.base : Int)) +
        (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
      tl.store(output_region + out0_bptr, (out0).to(DTYPE),
        mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
    }

def gridStrideCommuted : Parameters → ComputeKernel :=
  fun (c : Parameters) =>
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
        offset0 = $(c.B) * tile_id0
        in0_bptr = $((c.input.base : Int)) +
          (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.input.stride : Int))
        in0 = (tl.load(input_region + in0_bptr,
          mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n), dtype=DTYPE)).to(tl.float32)
        out0 = tl.extra.cuda.libdevice.pow($((c.base : ℝ)).to(tl.float32), in0)
        out0_bptr = $((c.output.base : Int)) +
          (offset0 + tl.arange(0, $(c.B))).to(tl.int64) * $((c.output.stride : Int))
        tl.store(output_region + out0_bptr, (out0).to(DTYPE),
          mask=(offset0 + tl.arange(0, $(c.B))) < $(c.n))
      }
    }

def kernel (c : Parameters) : ComputeKernel :=
  if c.oneTile then oneTileKernel c else gridStrideKernel c

def commutedKernel (c : Parameters) : ComputeKernel :=
  if c.oneTile then oneTileCommuted c else gridStrideCommuted c

end VeriTile.Bench.Repairs.PowScalarTensor.Checked.HistoricalBody

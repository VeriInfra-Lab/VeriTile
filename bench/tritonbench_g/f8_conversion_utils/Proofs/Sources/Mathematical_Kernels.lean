import bench.tritonbench_g.f8_conversion_utils.Kernels

/-! Compatibility interfaces refer directly to the canonical Python bodies.
The Real proofs erase precision from these same sources. No alternative
FP64 implementation or source-generation macro is retained. -/
namespace VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource
open VeriTile.Triton
abbrev toHalf := VeriTile.Bench.Repairs.F8ConversionUtils.Body.toHalf
abbrev toFP8 := VeriTile.Bench.Repairs.F8ConversionUtils.Body.toFP8

/-- Representative decode profile, consumed only after precision erasure. -/
def realForward (swapped : Bool) : sourceType := toHalf .e5 swapped
/-- Representative encode profile, consumed only after precision erasure. -/
def realReverse (swapped : Bool) : sourceType := toFP8 .fp16 .e5 swapped
end VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource

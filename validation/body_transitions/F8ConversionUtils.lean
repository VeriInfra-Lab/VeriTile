import bench.tritonbench_g.f8_conversion_utils.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.F8ConversionUtils.SourceTransition
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.F8ConversionUtils.TypedSource

theorem all_decode_programs (format : Format) (swapped : Bool)
    (Y X : RegionName) (N B : Nat) :
    toHalf format swapped Y X N B = Body.toHalf format swapped Y X N B := by
  cases format <;> cases swapped <;> rfl

theorem all_encode_programs (input : Input) (format : Format) (swapped : Bool)
    (Y X : RegionName) (N B : Nat) :
    toFP8 input format swapped Y X N B = Body.toFP8 input format swapped Y X N B := by
  cases input <;> cases format <;> cases swapped <;> rfl

#axiomsClean all_decode_programs
#axiomsClean all_encode_programs
end VeriTile.Bench.Repairs.F8ConversionUtils.SourceTransition

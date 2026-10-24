import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.f8_conversion_utils.Kernels
import bench.repairs.tritonbench_g.f8_conversion_utils.Proofs.LegacySource
namespace VeriTile.Bench.Repairs.F8ConversionUtils.SourcePreservation
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.F8ConversionUtils

def legacyFormat : TypedSource.Format → Legacy.Format | .e4 => .e4 | .e5 => .e5
def legacyInput : TypedSource.Input → Legacy.Input | .fp16 => .fp16 | .fp32 => .fp32

theorem toHalf_preserved (format : TypedSource.Format) (swapped : Bool) :
    TypedSource.toHalf format swapped = Legacy.toHalf (legacyFormat format) swapped := by
  cases format <;> cases swapped <;> rfl

theorem toFP8_preserved (input : TypedSource.Input) (format : TypedSource.Format) (swapped : Bool) :
    TypedSource.toFP8 input format swapped = Legacy.toFP8 (legacyInput input) (legacyFormat format) swapped := by
  cases input <;> cases format <;> cases swapped <;> rfl

/-- The existing FP64 erasure control keeps the duplicate decoder store. -/
theorem realForward_preserved (swapped : Bool) :
    TypedSource.realForward swapped =
      (if swapped then legacyF8Forward%[tl.float64,tl.float64,indexCommuted]
       else legacyF8Forward%[tl.float64,tl.float64,original]) := by
  cases swapped <;> rfl

/-- The existing FP64 reverse control keeps its single encoder store. -/
theorem realReverse_preserved (swapped : Bool) :
    TypedSource.realReverse swapped =
      (if swapped then legacyF8Reverse%[tl.float64,tl.float64,indexCommuted]
       else legacyF8Reverse%[tl.float64,tl.float64,original]) := by
  cases swapped <;> rfl

/-- Full original/candidate function families, all six native profiles and all
call arguments, match the existing complete source programs. -/
theorem pasted_toHalf (format : TypedSource.Format) (swapped : Bool) :
    Source.toHalf format swapped = TypedSource.toHalf format swapped := by
  cases format <;> cases swapped <;> rfl

theorem pasted_toFP8 (input : TypedSource.Input) (format : TypedSource.Format) (swapped : Bool) :
    Source.toFP8 input format swapped = TypedSource.toFP8 input format swapped := by
  cases input <;> cases format <;> cases swapped <;> rfl

#axiomsClean pasted_toHalf
#axiomsClean pasted_toFP8
#axiomsClean realForward_preserved
#axiomsClean realReverse_preserved
#axiomsClean toHalf_preserved
#axiomsClean toFP8_preserved
end VeriTile.Bench.Repairs.F8ConversionUtils.SourcePreservation

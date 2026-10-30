import bench.repairs.tritonbench_g.kv_cache_filling.Proofs.FP
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.KvCacheFilling.Counterexamples
open VeriTile.Triton

/-! These are deterministic facts about the existing binary32 model, not
experimental numerical assumptions. The Python input contract constrains
metadata and storage but does not inspect FP32 payload values. Consequently
the old `TaskCastDomain` premise does not follow from the admitted contract.
No guard or retained conditional theorem is changed by these diagnostics. -/

def zero : Word := FP.Value.zero .fp32
def tiny : Word := FP.Value.ofBits .fp32 1
def nan : Word := FP.Value.qNaN .fp32
def ops := softwareOps (fun _ => 0)

/-- Both operands of the underflow example are finite FP32 words. -/
theorem tiny_inputs_finite : zero.toRat?.isSome ∧ tiny.toRat?.isSome := by
  decide +kernel

/-- A nonconstant finite row `[0, 2^-149]` is not covered by the repair's
constant-row fallback. Its policy-8 scale rounds to zero. -/
theorem int8_scale_underflows : scaleAt ops 255 zero tiny = zero := by
  decide +kernel

/-- The same underflow occurs in the policy-4 row `[0, 2^-149]`. -/
theorem int4_scale_underflows : scaleAt ops 15 zero tiny = zero := by
  decide +kernel

/-- The first active lane then divides zero by zero and cannot be converted
using the shared defined-domain uint8 conversion. -/
theorem int8_active_cast_undefined :
    FP.IntegerConversion.toUnsigned 8 (roundedAt ops 255 zero tiny zero) = none := by
  decide +kernel

theorem int4_active_cast_undefined :
    FP.IntegerConversion.toUnsigned 8 (roundedAt ops 15 zero tiny zero) = none := by
  decide +kernel

/-- Admitting a NaN payload also makes the old defined-cast premise false. -/
theorem nan_active_cast_undefined :
    FP.IntegerConversion.toUnsigned 8 (roundedAt ops 255 nan nan nan) = none := by
  decide +kernel

/-- An arbitrary extension of undefined conversion cannot be discarded:
the resulting active byte really depends on that extension. -/
theorem int8_result_depends_on_undefined_cast :
    (softwareOps (fun _ => 0)).uint8 (roundedAt ops 255 zero tiny zero) ≠
      (softwareOps (fun _ => 1)).uint8 (roundedAt ops 255 zero tiny zero) := by
  simp only [softwareOps, FP.IntegerConversion.totalUnsigned,
    int8_active_cast_undefined, Option.getD_none]
  decide

#axiomsClean tiny_inputs_finite
#axiomsClean int8_scale_underflows
#axiomsClean int4_scale_underflows
#axiomsClean int8_active_cast_undefined
#axiomsClean int4_active_cast_undefined
#axiomsClean nan_active_cast_undefined
#axiomsClean int8_result_depends_on_undefined_cast

end VeriTile.Bench.Repairs.KvCacheFilling.Counterexamples

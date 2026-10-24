import bench.repairs.tritonbench_g.fp4_to_bf16.ScheduleIO
import bench.repairs.tritonbench_g.fp4_to_bf16.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Fp4ToBf16.ScheduleIO
open VeriTile Triton FP.Structural
open VeriTile.Bench.TritonBenchG.Fp4ToBf16 FPValue Host
open scoped VeriTile.Spec
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- Reuse the established complete-source conversion result for each existing
CTA order. No floating arithmetic is reassociated or replaced. -/
theorem equivalent (h : Input) (storage : Finite.Storage h.shape.count) (a b : List Nat)
    (fresh : h.input≠h.output) (gridSafe : grid h≤2147483647)
    (left : a.Perm (List.range (grid h))) (right : b.Perm (List.range (grid h))) :
    original h a ≡[assumptions] candidate h b := by
  apply Spec.FloatingPoint.ofStructural (structural := Equivalent)
    (lhs := original h a) (rhs := candidate h b) rfl rfl
  intro rest
  apply ProgramIO.EquivalentOn.of_implementsOn
    (ConversionIO.native_implements rest h storage a fresh gridSafe left)
  exact ConversionIO.native_implements rest h storage b fresh gridSafe right

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.Fp4ToBf16.ScheduleIO

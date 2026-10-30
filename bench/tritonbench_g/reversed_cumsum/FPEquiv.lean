import bench.tritonbench_g.reversed_cumsum.Proofs.FPPrograms
import bench.tritonbench_g.reversed_cumsum.Proofs.Rewrite

/-! The actual triangular-comparison variant of the complete repaired source.
No floating arithmetic, accumulation order or precision is changed. -/
namespace VeriTile.Bench.Repairs.ReversedCumsum.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section


specification reversed_cumsum_compare_right_equiv (h : Call) (c : Config) :
    original h c ≡[assumptions] compareRight h c := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original h c) (rhs := compareRight h c)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D _ u _
  exact Rewrite.whole_state (engine M) h c u


#print_fp_assumptions reversed_cumsum_compare_right_equiv
#axiomsClean reversed_cumsum_compare_right_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsum.Equivalence

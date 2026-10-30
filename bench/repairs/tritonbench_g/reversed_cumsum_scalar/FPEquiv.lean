import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPCompatibility

/-! The original index-commutation equivalence; floating operations and source
precision are unchanged. Execution and memory auxiliaries live in Proofs. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.Equivalence
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
noncomputable section

specification reversed_cumsum_scalar_equiv (h : Call) (c : Config) :
    original h c ≡[assumptions] indexCommuted h c := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := original h c) (rhs := indexCommuted h c)
    (structural := fun _ _ => False) rfl rfl
  intro α _ M D _ u _
  exact Rewrite.whole_state (engine M) h c u

#print_fp_assumptions reversed_cumsum_scalar_equiv
#stmtSurfaceSubset reversed_cumsum_scalar_equiv ⊆
  [Call,Config,Spec.FloatingPoint,Spec.Equiv,FP.GuardedFragment,FP.ProfiledRewrite.Program,
    FP.ProfiledRewrite.instProgramSyntaxProgram,
    original,indexCommuted,assumptions]
#axiomsClean reversed_cumsum_scalar_equiv
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.Equivalence

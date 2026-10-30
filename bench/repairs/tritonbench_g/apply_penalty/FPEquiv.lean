import bench.repairs.tritonbench_g.apply_penalty.Proofs.KernelResults

namespace VeriTile.Bench.Repairs.ApplyPenalty.FPEquiv
open VeriTile.Triton
open scoped VeriTile.Spec
noncomputable section

/-- The existing division and reciprocal FP32 source programs. Their program
view retains the active positive-arm domain and quantifies every program id. -/
abbrev original := RawFP.original
abbrev reciprocal := RawFP.reciprocal

/-- The actual reciprocal transformation, conditional on the existing admitted
DIV-MUL-RCP rule and precisely its active positive-arm input conditions. Both
complete sources succeed and agree on typed memory and the exterior frame. -/
specification apply_penalty_reciprocal_equiv (stride B : Nat) (R : FP.ScalarArithmetic.Rules) :
    original stride B ≡[R] reciprocal stride B :=
  RawFP.apply_penalty_reciprocal_equiv stride B R

#stmtSurfaceSubset apply_penalty_reciprocal_equiv ⊆
  [original,reciprocal,FP.ScalarArithmetic.Rules,FP.ScalarArithmetic.Rules.assumptions,
   Spec.Equiv,Rewrite.Program,Rewrite.instProgramSyntaxProgram]
#print_fp_assumptions apply_penalty_reciprocal_equiv
#axiomsClean apply_penalty_reciprocal_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ApplyPenalty.FPEquiv

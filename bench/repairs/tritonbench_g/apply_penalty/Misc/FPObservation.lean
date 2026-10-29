import bench.repairs.tritonbench_g.apply_penalty.IO
import bench.repairs.tritonbench_g.apply_penalty.Proofs.PublicEquivalence

namespace VeriTile.Bench.Repairs.ApplyPenalty.FPEquiv
open VeriTile.Triton FP.Structural FP.Guarded PublicIO
noncomputable section

/-- Retained full IO application: both actual source traces are safe, both
executions succeed, and every active output and exterior cell agrees. -/
theorem apply_penalty_reciprocal_io_equiv {α : Type} [Inhabited α]
    (R : FP.ScalarArithmetic.Rules) (M : FP.Structural.Algebra α) (D : Domain α)
    (hM : Models R.assumptions M D) (lt : α→α→Bool)
    (hlt : M.compareLt (some .fp32) .real=some lt) (stride B pid : Nat) :
    (io M stride B pid).EquivalentOn (candidate M stride B pid) (DivisionDomain M D lt) :=
  equivalent R M D hM lt hlt stride B pid

#stmtSurfaceSubset apply_penalty_reciprocal_io_equiv ⊆
  [FP.ScalarArithmetic.Rules,FP.Structural.Algebra,Domain,Models,
   FP.ScalarArithmetic.Rules.assumptions,FP.Structural.Algebra.compareLt,
   ComputeDType,ComputeDType.fp32,FloatDType.real,TileDType.real,io,candidate,DivisionDomain,ProgramIO.EquivalentOn]
#axiomsClean apply_penalty_reciprocal_io_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.ApplyPenalty.FPEquiv

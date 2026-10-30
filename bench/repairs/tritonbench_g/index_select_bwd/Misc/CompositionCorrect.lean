import bench.repairs.tritonbench_g.index_select_bwd.Correct

namespace VeriTile.Bench.Repairs.IndexSelectBwd.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- Complete ordered source composition: every incoming row and column slice,
including repeated destination indices and nonzero initial gradients. -/
theorem index_select_bwd_complete_real (rest : Algebra ℝ) {R N C : Nat}
    (index : Fin N → Fin R) (current : Logical.Coord R C → ℝ)
    (incoming : Logical.Coord N C → ℝ) (q : Logical.Coord R C) :
    Spec.Real (Program.run (KernelProof.arithmetic rest (· + ·)) index incoming current q =
      Logical.target index current incoming q) := by
  rw [Program.run_correct]
  exact Logical.complete_real index current incoming q

#specNonCircular Logical.target avoiding [kernel, Source.execution, Program.run]
#axiomsClean index_select_bwd_complete_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.IndexSelectBwd.Correct

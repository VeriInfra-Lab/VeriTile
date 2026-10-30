import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.TypedFinish

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural Kernels
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

def initialPointer (v : Bool) (count cols : Nat) : ComputeStmt :=
  ((kernel v "input" "output" count cols).surfaceBody[2]?).getD (.forRange "" 0 0 1 [])

theorem pointer_step {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α) :
    step M (initialPointer Bool.false count cols) s =
      step M (initialPointer Bool.true count cols) s := by
  simp only [initialPointer, kernel, ComputeKernel.surfaceBody,
    List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some,
    Bool.false_eq_true, ↓reduceIte]
  simp [step, evalExpr, evalComputeOp, evalOp_unfold, numeric]
  unfold FP.Structural.bop
  simp only [Broadcast.leftIndex, Broadcast.rightIndex, Nat.mul_comm]

theorem variant_decomposition (v : Bool) (count cols : Nat) :
    (kernel v "input" "output" count cols).surfaceBody =
      (source count cols).surfaceBody.take 2 ++ [initialPointer v count cols] ++
      (source count cols).surfaceBody.drop 3 := by cases v <;> rfl

/-- Whole optional-state equality for the genuine initial-multiplication
commutation. No arithmetic is performed on payload words. -/
theorem source_equivalence {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α) :
    exec M (source count cols) s = exec M (commuted count cols) s := by
  change run M (kernel Bool.false "input" "output" count cols).surfaceBody s =
    run M (kernel Bool.true "input" "output" count cols).surfaceBody s
  conv_lhs => rw [variant_decomposition Bool.false]
  conv_rhs => rw [variant_decomposition Bool.true]
  simp only [run_append, run, pointer_step]

theorem candidate_run {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α) :
    exec M (commuted count cols) s = some (finalState count cols s) := by
  rw [← source_equivalence]
  exact source_run M count cols s

end VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms

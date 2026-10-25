import bench.repairs.tritonbench_g.nested_loops_processing.Proofs.SignedMemory

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution
open VeriTile.Triton FP.Structural Kernels
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- The final state of the current raw signed source, including its actual registers. -/
abbrev actualFinalState {α : Type} [Inhabited α] (count cols : Nat) (s : State α) : State α :=
  SignedExecution.finalState count cols s

theorem source_run {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α) :
    exec M (source count cols) s = some (actualFinalState count cols s) :=
  SignedExecution.source_run M count cols s

/-- Complete memory connection to the independently proved mathematical skeleton. -/
theorem source_memory {α : Type} [Inhabited α] (count cols : Nat) (s : State α) :
    (actualFinalState count cols s).mem = (finalState count cols s).mem :=
  SignedExecution.finalState_memory count cols s

theorem source_equivalence {A : Type} [Inhabited A] (M : Algebra A) (count cols : Nat) (s : State A) :
    exec M (source count cols) s = exec M (commuted count cols) s := by
  have first : run M ((source count cols).surfaceBody.take 6) s =
      run M ((commuted count cols).surfaceBody.take 6) s := by
    simp [source,commuted,kernel,originalKernel,commutedKernel,quotedOriginal,quotedCommuted,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      numeric,bop,ComputeDType.eraseDType,Option.bind]
    congr 1
    funext i
    simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Int.mul_comm]
  change run M (source count cols).surfaceBody s = run M (commuted count cols).surfaceBody s
  rw [←List.take_append_drop 6 (source count cols).surfaceBody,
    ←List.take_append_drop 6 (commuted count cols).surfaceBody,run_append,run_append,first]
  rfl

theorem candidate_run {α : Type} [Inhabited α] (M : Algebra α)
    (count cols : Nat) (s : State α) :
    exec M (commuted count cols) s = some (actualFinalState count cols s) := by
  rw [←source_equivalence]
  exact source_run M count cols s

#axiomsClean source_run
#axiomsClean source_memory
#axiomsClean source_equivalence
#axiomsClean candidate_run
end VeriTile.Bench.Repairs.NestedLoopsProcessing.TypedExecution

#guard_msgs (drop info) in
#auditModuleAxioms

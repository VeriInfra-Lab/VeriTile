import bench.repairs.tritonbench_g.dequantize_matmul.Proofs.KernelExecution

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelEquivalence
open VeriTile.Triton FP.Structural KernelSource KernelExecution
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

private abbrev bodyRun {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) :=
  FP.Structural.run M

private theorem run_append {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (xs ys : List ComputeStmt) (s : State α) :
    bodyRun M p (xs ++ ys) s =
      (bodyRun M p xs s).bind (bodyRun M p ys) := by
  exact FP.Structural.run_append M xs ys s

/-- Source statement 7 is the sole N-mask product rewrite. This extracts
that statement from the actual quotation; no second AST is maintained. -/
private def addressBody (p : Parameters) (version : Bool) : List ComputeStmt :=
  ((kernel version p).surfaceBody.drop 7).take 1

private theorem address_run {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (s : State α) :
    bodyRun M p (addressBody p Bool.false) s =
      bodyRun M p (addressBody p Bool.true) s := by
  cases hn : s.regs .int [] "n_block_idx" <;>
    cases ho : s.regs .nat [p.BN] "offs_n" <;>
    simp (config := { maxSteps := 1000000 }) [bodyRun, addressBody, kernel, original,
      candidate, TypedSource.kernel, ComputeKernel.surfaceBody, List.drop, List.take,
      FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
      FP.Structural.evalOp_unfold, FP.Structural.numeric, FP.Structural.numericLt,
      FP.Structural.bop, Broadcast.leftIndex, Broadcast.rightIndex,
      ofFloat, toFloat, Region.cast, hn, ho, Option.bind, Int.mul_comm]
  all_goals try (congr 1; funext i; simp [FP.Structural.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, Int.mul_comm])

private theorem split_source (p : Parameters) (version : Bool) :
    (kernel version p).surfaceBody =
      (kernel version p).surfaceBody.take 7 ++
        (addressBody p version ++ (kernel version p).surfaceBody.drop 8) := by
  cases version <;> rfl

private theorem run_split {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (version : Bool) (s : State α) :
    exec M p version s =
      (bodyRun M p ((kernel version p).surfaceBody.take 7) s).bind
        (bodyRun M p (addressBody p version ++ (kernel version p).surfaceBody.drop 8)) := by
  change bodyRun M p (kernel version p).surfaceBody s = _
  conv_lhs => rw [split_source p version]
  exact run_append M p _ _ s

theorem same_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    exec M p Bool.false s = exec M p Bool.true s := by
  have hp : (kernel Bool.false p).surfaceBody.take 7 =
      (kernel Bool.true p).surfaceBody.take 7 := rfl
  have ht : (kernel Bool.false p).surfaceBody.drop 8 =
      (kernel Bool.true p).surfaceBody.drop 8 := rfl
  have hs : bodyRun M p (addressBody p Bool.false ++ (kernel Bool.false p).surfaceBody.drop 8) =
      bodyRun M p (addressBody p Bool.true ++ (kernel Bool.true p).surfaceBody.drop 8) := by
    funext next
    rw [run_append M p (addressBody p Bool.false) _ next,
      run_append M p (addressBody p Bool.true) _ next, ht, address_run]
  rw [run_split M p Bool.false s, run_split M p Bool.true s, hp, hs]

#axiomsClean same_run
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelEquivalence

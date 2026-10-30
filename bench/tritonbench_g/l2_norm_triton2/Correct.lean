import bench.tritonbench_g.l2_norm_triton2.Proofs.LogicalIO

/-! Two genuine kernels, each with its complete row grid. Public channels,
windows, masks and the runtime epsilon ABI are data in IO.lean. Raw execution,
physical allocations and Tensor binding facts use helpers in Proofs/ and
optional results in Misc/TensorBridge.lean outside these mathematical specifications. -/
namespace VeriTile.Bench.Repairs.L2NormTriton2.Correct
open VeriTile.Triton FP.Structural
open scoped VeriTile.Triton.FP.Structural.LogicalProgramIO
noncomputable section

/-- Every row implements x / sqrt(sum x² + eps), with actual source access
safety, typed outputs and the complete exterior memory frame. -/
specification forward_correctness (rows N B : Nat) (cover : N≤B) (eps : ℝ) :
    Spec.Real (L2NormTriton1.LogicalIO.io L2NormTriton1.RealExecution.algebra rows N B eps ⊨
      fun x row col => x row col / Real.sqrt ((∑ j : Fin N, (x row j)^2)+eps)) :=
  L2NormTriton1.LogicalIO.implements rows N B cover eps

/-- Every row implements the independent full L2 gradient. Real sqrt and
division are total; no numerical-value restriction is added to epsilon. -/
specification backward_correctness (rows N B : Nat) (cover : N≤B) (eps : ℝ) :
    Spec.Real (LogicalIO.io RealExecution.algebra rows N B eps ⊨ fun a row col =>
      let q := (∑ j : Fin N, (a.x row j)^2)+eps
      let dot := ∑ j : Fin N, a.dy row j*a.x row j
      a.dy row col / Real.sqrt q-dot/q/Real.sqrt q*a.x row col) :=
  LogicalIO.implements rows N B cover eps

#axiomsClean forward_correctness
#axiomsClean backward_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.Correct

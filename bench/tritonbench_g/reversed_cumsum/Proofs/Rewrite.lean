import bench.tritonbench_g.reversed_cumsum.Proofs.FPProgram
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ReversedCumsum.Rewrite
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3600000
set_option maxRecDepth 24000

theorem prefix_run {α : Type} [Inhabited α] (M : Algebra α)
    (h : Call) (c : Config) (u : State α) :
    run M ((Kernels.source h c).surfaceBody.take 4) u =
      run M ((Kernels.compareRight h c).surfaceBody.take 4) u := by
  simp [Kernels.source,Kernels.compareRight,ComputeKernel.surfaceBody,
    run,step,FP.Structural.evalExpr,evalOp_unfold,numericLe,natLe,
    bop,Broadcast.leftIndex,Broadcast.rightIndex,TileShape.dropInsertedIndex,bind,Option.bind]

theorem common_suffix (h : Call) (c : Config) :
    (Kernels.source h c).surfaceBody.drop 4 =
      (Kernels.compareRight h c).surfaceBody.drop 4 := rfl

/-- Reverse the operands of the natural triangular comparison only. The FP32
dot, carry, reductions, casts, masks and stores retain their complete context. -/
theorem whole_state {α : Type} [Inhabited α] (M : Algebra α)
    (h : Call) (c : Config) (u : State α) :
    FP.Structural.exec M (Kernels.source h c) u =
      FP.Structural.exec M (Kernels.compareRight h c) u := by
  change run M (Kernels.source h c).surfaceBody u = run M (Kernels.compareRight h c).surfaceBody u
  conv_lhs => rw [←List.take_append_drop 4 (Kernels.source h c).surfaceBody]
  conv_rhs => rw [←List.take_append_drop 4 (Kernels.compareRight h c).surfaceBody]
  simp only [run_append,prefix_run,common_suffix]

#axiomsClean prefix_run
#axiomsClean whole_state
end
end VeriTile.Bench.Repairs.ReversedCumsum.Rewrite

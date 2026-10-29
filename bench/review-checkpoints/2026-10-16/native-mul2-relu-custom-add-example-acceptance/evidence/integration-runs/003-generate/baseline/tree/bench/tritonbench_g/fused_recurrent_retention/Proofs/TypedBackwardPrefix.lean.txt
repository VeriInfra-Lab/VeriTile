import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedBackwardReady

/-! Complete typed backward header and seed branch execute from arbitrary
registers and memory, with the seed flag still symbolic. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem prefix_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout)
    (r : BackwardRegions) (hasSeed : Bool) (u : State α) :
    ∃ v, run A (Program.backwardFirst Bool.false .fp32 .fp32 p r hasSeed).before u =
      some v ∧ FirstReady p v := by
  cases hasSeed <;>
    simp [Program.backwardFirst,Program.backward,TypedSource.backward,TypedSource.backwardOriginal,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralMod,Region.cast,FirstReady]

#axiomsClean prefix_success
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedBackward

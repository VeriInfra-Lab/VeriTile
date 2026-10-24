import bench.tritonbench_g.fused_recurrent_retention.Proofs.TypedModel

/-! The complete fp32 source header and optional seed branch execute from
arbitrary initial typed registers. Both flags remain symbolic. -/
namespace VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

theorem prefix_success {α : Type} [Inhabited α] (A : Algebra α) (p : Layout) (r : ForwardRegions) (hasSeed storeFinal : Bool)
    (u : State α) :
    ∃ v, run A (Program.forwardParts Bool.false .fp32 .fp32 p r hasSeed storeFinal).before u =
      some v ∧ Ready p v := by
  cases hasSeed <;>
    simp [Program.forwardParts,Program.forward,TypedSource.forward,TypedSource.forwardOriginal,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,FP.Structural.evalExpr,
      evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,numeric,bop,numericLt,
      natLt,integralMod,Region.cast,Ready]

#axiomsClean prefix_success
end
end VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full.TypedExecution

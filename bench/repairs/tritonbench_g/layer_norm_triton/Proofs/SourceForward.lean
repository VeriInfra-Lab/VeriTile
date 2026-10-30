import bench.repairs.tritonbench_g.layer_norm_triton.Proofs.SourceModel

namespace VeriTile.Bench.Repairs.LayerNormTriton.SourceForward
open VeriTile.Triton FP.Structural SourceModel
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem source_executes [Inhabited α] (m : Model α) (stride N B : Nat) (eps : ℝ)
    (positive : 0<N) (covers : N≤B) (s : State α) :
    ∃ t,FP.Structural.exec (engine m)
      (TypedSource.forward "X" "Y" "W" "Bias" "Mean" "Rstd" stride N B eps) s=some t := by
  simp [FP.Structural.exec,TypedSource.forward,ComputeKernel.surfaceBody,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,
    range_one _ _ N B positive covers,
    engine,numeric,natLt,numericLt,numericLe,numericEq,bop,
    State.setReg,FP.Structural.store,ComputeDType.eraseDType,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    Region.cast,Option.bind,ofFloat,toFloat]

#axiomsClean source_executes
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.LayerNormTriton.SourceForward

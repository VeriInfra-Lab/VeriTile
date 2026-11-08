import bench.tritonbench_g.attention_fwd_triton3.Proofs.BodyStages

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem update_phase (c : Call) (o : Ops α) (positive : 0<c.BN)
    (initialState s : State α) (old : Carry c α) (cur : Int)
    (ready : ScoresReady c o (inputs initialState) cur old s)
    (memory : s.mem=initialState.mem) :
    ∃ t,run (engine o.algebra) (((parts c).body.drop 7).take 12) s=some t ∧
      UpdateReady c o (inputs initialState) cur old t ∧ t.mem=initialState.mem := by
  rcases ready with ⟨live,hscore,hmask,hcounter⟩
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16⟩
  by_cases he : c.params.evenN=0 <;> by_cases hw : c.window=0
  all_goals
    simp (config := {maxSteps := 250000}) [parts,Kernels.source,Quoted.kernel,DSL.CallValue.natural,DSL.CallValue.integer,
      DSL.CallValue.fp32,sourceKernel,ComputeKernel.withScalarArguments,
      ScalarArgument.statement,ScalarArgument.fp32,ComputeKernel.surfaceBody,Source.intRange?,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,ComputeDType.eraseDType,
      State.setReg,engine,Algebra.withDefaultPrecision,FPBackend.Ops.algebra,resolvePrecision,
      numeric,numericLt,numericLe,numericEq,natLt,natLe,integralFloorDiv,integralMod,
      ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,
      TileShape.axisDim,positive,Matches,inputs,memory,bop,Broadcast.leftIndex,Broadcast.rightIndex,
      hscore,hmask,hcounter,he,hw,UpdateReady,scaledCarry,alphas,probabilities,nextCenter]
  all_goals simp [funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference

import bench.tritonbench_g.attention_fwd_triton3.Proofs.CSEStages

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem score_phase (c : Call) (o : Ops α) (positive : 0<c.BN)
    (initialState s : State α) (old : Carry c α)
    (live : Matches c o (inputs initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) :
    ∃ t,run (engine o.algebra) ((parts c).body.take 7)
        (s.setReg "__veritile_statement_jit_local_11_start_n" .int [] (fun _=>cur))=some t ∧
      ScoresReady c o (inputs initialState) cur old t ∧ t.mem=initialState.mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16⟩
  by_cases he : c.params.evenN=0 <;> by_cases hw : c.window=0 <;> by_cases hc : c.complement=0
  all_goals
    simp (config := {maxSteps := 1000000}) [parts,Kernels.keyMaskCSE,Quoted.keyMaskCSE,
      DSL.CallValue.natural,DSL.CallValue.integer,DSL.CallValue.fp32,
      kernelCSEKernel,kernelCSEKernel,ComputeKernel.withScalarArguments,ScalarArgument.statement,ScalarArgument.fp32,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,State.setReg,engine,Algebra.withDefaultPrecision,
      FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
      natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,he,hw,hc,
      TileShape.axisDim,positive,ScoresReady,Reference.ScoresReady,Matches,scores,loadBlock,windowMask,memory,inputs,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simp [loadBlock,windowMask,inputs,he,hw,hc,funext_iff,bop,
      Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,Int.natCast_mul]
  all_goals repeat' constructor
  all_goals intros; rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE

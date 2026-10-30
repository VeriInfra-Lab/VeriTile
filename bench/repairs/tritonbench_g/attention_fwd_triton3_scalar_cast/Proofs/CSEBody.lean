import bench.repairs.tritonbench_g.attention_fwd_triton3_scalar_cast.Proofs.CSEParts

namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Reference
open VeriTile.Bench.Repairs.AttentionFwdTriton3.Host
open VeriTile.Bench.Repairs.ContextAttnMistral
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem body (c : Call) (o : Ops α) (positive : 0<c.BN)
    (initialState s : State α) (old : Carry c α)
    (live : Matches c o (inputs initialState) old s)
    (memory : s.mem=initialState.mem) (cur : Int) :
    ∃ t,run (VeriTile.Triton.FP.ProfiledRewrite.engine o.algebra) (parts c).body
        (s.setReg "start_n" .int [] (fun _ => cur))=some t ∧
      Matches c o (inputs initialState) (advance c o (inputs initialState) cur old) t ∧
      t.mem=initialState.mem := by
  rcases live with ⟨h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12⟩
  by_cases he : c.params.evenN=0 <;> by_cases hw : c.window=0 <;> by_cases hc : c.complement=0
  all_goals
    simp (config := {maxSteps := 1000000}) [parts,Kernels.keyMaskCSE,Quoted.keyMaskCSE,
      ComputeKernel.surfaceBody,Source.intRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,State.setReg,VeriTile.Triton.FP.ProfiledRewrite.engine,Algebra.withDefaultPrecision,
      FPBackend.Ops.algebra,resolvePrecision,numeric,numericLt,numericLe,numericEq,
      natLt,natLe,integralFloorDiv,integralMod,ofFloat,toFloat,Region.cast,boundaryZero,
      h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,he,hw,hc,
      TileShape.axisDim,positive,Matches,advance,loadBlock,windowMask,memory,inputs,
      bop,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals
    simp [loadBlock,funext_iff,bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals refine ⟨?_,?_,?_⟩ <;> intros
  all_goals rfl

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.KeyMaskCSE

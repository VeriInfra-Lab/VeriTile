import bench.tritonbench_g.attn_fwd_causal.Proofs.Context
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Complete source iteration, including both typed dots, tail mask, explicit
half conversions and all three pointer advances. -/
def bodyExec {phase : Bool} (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current (phase:=phase)) :
    {t : State α // run (engine (withDot M dot)) (loopParts p phase).body
      (current.setReg (workName phase "start_n") .nat [] (fun _ => row))=some t} := by
  cases phase
  all_goals
    refine ⟨?_,?_⟩
    swap
    have h_length:=ctx.length
    have h_startM:=ctx.startM
    have h_firstLo:=ctx.firstLo
    have h_firstHi:=ctx.firstHi
    have h_rows:=ctx.rows
    have h_cols:=ctx.cols
    have h_query:=ctx.query
    have h_scale:=ctx.scale
    have h_keyPointer:=ctx.keyPointer
    have h_valuePointer:=ctx.valuePointer
    have h_scalePointer:=ctx.scalePointer
    have h_maximum:=ctx.maximum
    have h_normalizer:=ctx.normalizer
    have h_value:=ctx.value
    simp [workName] at h_length h_startM h_firstLo h_firstHi h_rows h_cols h_query h_scale h_keyPointer h_valuePointer h_scalePointer h_maximum h_normalizer h_value
    simp [loopParts,diagonal,parts,workName,kernel,TypedSource.repairedKernel,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    h_length,ctx.memory,h_startM,h_firstLo,h_firstHi,ctx.keyBase,ctx.valueBase,ctx.scaleBase,h_rows,h_cols,h_query,h_scale,h_keyPointer,h_valuePointer,
    h_scalePointer,ctx.outputPointer,h_maximum,h_normalizer,h_value]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source

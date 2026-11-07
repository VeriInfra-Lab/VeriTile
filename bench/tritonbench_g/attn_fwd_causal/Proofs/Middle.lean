import bench.tritonbench_g.attn_fwd_causal.Proofs.Values
namespace VeriTile.Bench.Repairs.AttnFwdCausal.Source
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def middleExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (middle p) current=some t} := by
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
  refine ⟨?_,?_⟩
  swap
  simp [middle,parts,kernel,TypedSource.repairedKernel,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.callerStartM,ctx.callerRows,ctx.callerCols,ctx.callerQuery,ctx.callerScale,h_maximum,h_normalizer,h_value,h_startM,ctx.keyBase,ctx.valueBase,ctx.scaleBase]
  rfl

private theorem minimum_if (x y : Nat) : (if x<y then x else y)=min x y := by
  by_cases h : x<y
  · simp only [h,ite_true]
    exact (Nat.min_eq_left (Nat.le_of_lt h)).symm
  · simp only [h,ite_false]
    exact (Nat.min_eq_right (by omega)).symm

theorem middle_context (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    Context M p initial (initial.pids 0*128) a (middleExec M dot p initial current row a ctx).val (phase:=Bool.true) := by
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
  constructor
  all_goals try simp [middleExec,workName,stop,h_length,ctx.callerStartM,ctx.callerRows,ctx.callerCols,ctx.callerQuery,ctx.callerScale,ctx.memory,h_startM,h_firstLo,h_firstHi,ctx.keyBase,ctx.valueBase,ctx.scaleBase,
    h_rows,h_cols,h_query,h_scale,ctx.outputPointer,h_maximum,h_normalizer,h_value,
    Parameters.q,Parameters.k,Parameters.v,Parameters.o,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals try funext i
  all_goals try simp [Parameters.q,Parameters.k,Parameters.v,Parameters.o,
    Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
  all_goals simp [FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
    Nat.mul_mod,Nat.add_mul,Nat.mul_add,Nat.mul_assoc,Nat.mul_comm,Nat.mul_left_comm,
    Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]

  all_goals try ring
  all_goals exact minimum_if _ _

#axiomsClean middle_context
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source

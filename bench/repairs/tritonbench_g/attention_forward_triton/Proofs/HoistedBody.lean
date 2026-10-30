import bench.repairs.tritonbench_g.attention_forward_triton.Proofs.HoistedSource

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite Source
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Complete source iteration, including both typed dots, tail mask, explicit
half conversions and all three pointer advances. -/
def bodyExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).body
      (current.setReg "start_n" .nat [] (fun _ => row))=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,TypedSource.hoistedHeadMask,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.headMask,ctx.memory,ctx.rows,ctx.cols,ctx.query,ctx.scale,ctx.keyPointer,ctx.valuePointer,
    ctx.scalePointer,ctx.outputPointer,ctx.maximum,ctx.normalizer,ctx.value]
  rfl

theorem body_context (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    Context M p initial (row+64) (update M dot p initial row a)
      (bodyExec M dot p initial current row a ctx).val := by
  have row_step : (row+64)/64=row/64+1 := by omega
  refine ⟨?_,?_⟩
  · constructor
    all_goals simp [bodyExec,ctx.memory,ctx.aligned,ctx.rows,ctx.cols,ctx.query,ctx.scale,
      ctx.keyPointer,ctx.valuePointer,ctx.scalePointer,ctx.outputPointer,ctx.maximum,ctx.normalizer,ctx.value,
      update,maximum,rescale,probability,scores,contribution,key,keyScale,value,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
      Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,row_step]
    all_goals try funext i
    all_goals simp [update,maximum,rescale,probability,scores,contribution,key,keyScale,value,zero,
      FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def,
      Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm,row_step]
    all_goals try delta scores maximum rescale probability contribution key keyScale value zero
    all_goals try simp [Nat.add_mul,Nat.add_assoc,Nat.add_comm,Nat.add_left_comm]
    all_goals norm_num
    all_goals try omega
    all_goals try rfl

  · simp [bodyExec,ctx.headMask]

#axiomsClean bodyExec
#axiomsClean body_context
end
end VeriTile.Bench.Repairs.AttentionForwardTriton.HoistedSource

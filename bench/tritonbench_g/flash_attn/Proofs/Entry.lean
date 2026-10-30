import bench.tritonbench_g.flash_attn.Proofs.ExecutionModel

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def entry (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α) (s : State α) :
    {t : State α // run (engine (withDot M dot)) (parts p).before (bindScale scale s)=some t} := by
  rcases p with ⟨qbs,qh,qr,qd,kbs,kh,kr,kd,vbs,vh,vr,vd,obs,oh,orr,od,bs,heads,len,dim,causal⟩
  cases causal
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (maxSteps := 1000000) [parts,kernel,declared,ComputeKernel.surfaceBody,
      Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,bindScale]
    rfl

theorem entry_context (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α) (s : State α) :
    Context M p s scale 0 (initialAcc M p) (entry M dot p scale s).val := by
  rcases p with ⟨qbs,qh,qr,qd,kbs,kh,kr,kd,vbs,vh,vr,vd,obs,oh,orr,od,bs,heads,len,dim,causal⟩
  cases causal
  all_goals constructor
  all_goals try simp [entry,query,base,zero,initialAcc,qptr,kptr,vptr,ofFloat,toFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try funext i
  all_goals try simp [query,base,zero,initialAcc,qptr,kptr,vptr,ofFloat,toFloat,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex,Function.comp_def]
  all_goals try rfl

#axiomsClean entry_context
end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource

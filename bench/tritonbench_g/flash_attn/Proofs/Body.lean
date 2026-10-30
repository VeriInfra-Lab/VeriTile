import bench.tritonbench_g.flash_attn.Proofs.Entry

namespace VeriTile.Bench.Repairs.FlashAttn.ActualSource
open VeriTile Triton FP.Structural FP.ProfiledRewrite
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 24000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Execute every statement of one actual source iteration. Floating values
are arbitrary; the only private optional primitive adapter is total. -/
def bodyExec (M : Algebra α) (dot : Dot α) (p : Parameters) (scale : α)
    (initial current : State α) (row : Nat) (a : Acc p α)
    (ctx : Context M p initial scale row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).body
      (current.setReg "start_n" .nat [] (fun _=>row))=some t} := by
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
      TileShape.axisDim,Region.cast,State.setReg,
      ctx.memory,ctx.startM,ctx.head,ctx.rows,ctx.columns,ctx.outputOffset,ctx.query,
      ctx.keyPointer,ctx.valuePointer,ctx.maximum,ctx.normalizer,ctx.output]
    rfl

end
end VeriTile.Bench.Repairs.FlashAttn.ActualSource

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
def bodyExec (M : Algebra α) (dot : Dot α) (p : Parameters)
    (initial current : State α) (row : Nat) (a : Acc α)
    (ctx : Context M p initial row a current) :
    {t : State α // run (engine (withDot M dot)) (parts p).body
      (current.setReg "start_n" .nat [] (fun _ => row))=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [parts,kernel,TypedSource.repaired,
    ComputeKernel.surfaceBody,Triton.Source.dynamicRange?,Triton.Source.prefixBefore,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,
    ctx.memory,ctx.startM,ctx.firstLo,ctx.firstHi,ctx.keyBase,ctx.valueBase,ctx.scaleBase,ctx.rows,ctx.cols,ctx.query,ctx.scale,ctx.keyPointer,ctx.valuePointer,
    ctx.scalePointer,ctx.outputPointer,ctx.maximum,ctx.normalizer,ctx.value]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttnFwdCausal.Source

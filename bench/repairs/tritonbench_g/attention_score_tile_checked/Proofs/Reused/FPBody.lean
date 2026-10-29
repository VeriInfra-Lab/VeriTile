import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.FPContext

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPBody
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPEngine FPContext
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- The complete actual typed iteration executes from the established pointer,
key, scale and accumulator invariant. No successful-body premise is assumed. -/
def executes (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters)
    (initial current : State α) (row : Nat) (acc : Fin B → α)
    (ctx : Context M B D p initial row acc current) :
    {t : State α // run (engine (withDot M dot)) (parts B D p).body
      (current.setReg "start_m" .nat [] (fun _ => row))=some t} := by
  by_cases hm : p.evenM = Bool.true
  all_goals by_cases hw : p.window = Bool.true
  all_goals by_cases hc : p.complement = Bool.true
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,RealSource.source,TypedSource.original,hm,hw,hc,
      ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      ctx.memory,ctx.startN,ctx.query,ctx.normalizer,ctx.key,ctx.scale,ctx.output]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPBody

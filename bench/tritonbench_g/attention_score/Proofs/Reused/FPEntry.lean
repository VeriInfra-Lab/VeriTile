import bench.tritonbench_g.attention_score.Proofs.Reused.FPEngine

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPEntry
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPEngine
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 16000000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

/-- Actual typed prelude execution includes grouped-head address arithmetic,
both K load modes, FP32 scale setup and the repaired query-length loop bound. -/
def executes (M : Algebra α) (dot : Dot α) (B D : Nat) (p : RealSource.Parameters) (s : State α) :
    {t : State α // run (engine (withDot M dot)) (parts B D p).before s=some t} := by
  by_cases hn : p.evenN = Bool.true
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [parts,kernel,RealSource.source,TypedSource.original,hn,
      ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore,Option.map,Option.getD,
      run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg]
    rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.FPEntry

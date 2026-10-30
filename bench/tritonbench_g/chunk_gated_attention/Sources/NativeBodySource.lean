import bench.tritonbench_g.chunk_gated_attention.Sources.NativeEntry

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def bodyExecution (c : Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (chunk : Nat) (acc : HIndex c→NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc) :
    {t : State NativeGraph.Expr // run (engine algebra) (parts c base gateK hasSeed storeFinal).body
      (current.setReg "i_t" .nat [] (fun _=>chunk))=some t} := by
  have hacc:=ha
  unfold HasAcc at hacc
  cases gateK
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp (config := {maxSteps:=150000}) [parts,source,NativeSource.hOriginal,NativeSource.hSource,
      ComputeKernel.surfaceBody,Source.dynamicRange?,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,engine,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,TileShape.axisDim,Region.cast,
      State.setReg,integralFloorDiv,integralMod,ctx.bh,ctx.ik,ctx.iv,hacc,FP.Structural.store,
      algebra,ScalarGraph.algebra,NativeGraph.algebra]
    rfl

#axiomsClean bodyExecution
end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeValue

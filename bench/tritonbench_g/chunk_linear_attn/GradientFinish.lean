import bench.tritonbench_g.chunk_linear_attn.GradientBody
namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_memory)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def gradientFinish (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (heads : Nat) (input : Memory α) (s : State α) (a : GradAcc p α)
    (ctx : GradientReady p pid heads input s) (fixed : GradientFixed M D p pid input scale s)
    (ha : HasGrad p a s) :
    {t : State α // run (engine (withDot M D)) (gradientParts p scale).after s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,transposeValues,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList,ctx.ik,ctx.it,ctx.bh,ctx.oi,fixed.query,fixed.key,fixed.score,
    ha.query,ha.key,ha.score,FP.Structural.store]
  rfl

theorem gradientFinish_memory (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (pid : PID) (heads : Nat) (input : Memory α) (s : State α) (a : GradAcc p α)
    (ctx : GradientReady p pid heads input s) (fixed : GradientFixed M D p pid input scale s)
    (ha : HasGrad p a s) :
    (gradientFinish M D p scale pid heads input s a ctx fixed ha).val.mem=
      gradKeyWrite M p pid "keyGradient" (gradKeyResult M D p pid input scale a)
        (gradKeyWrite M p pid "queryGradient" (gradQueryResult M D p pid input scale a) s.mem) := by
  simp [gradientFinish,gradKeyWrite,scatter,masked_scatter_memory,gradKeyResult,gradQueryResult,
    gradScoreGradient,gradKeyActive,gradKeyOffset,transpose,zero,cast,add,mul,scalar,dot,transposeValues,
    FP.Structural.bop,Broadcast.leftIndex,Broadcast.rightIndex]
  unfold VeriTile.Bench.Repairs.ChunkDeltaFwd.scatter
  unfold gradKeyActive gradKeyOffset
  rfl

#axiomsClean gradientFinish_memory
end
end VeriTile.Bench.Repairs.ChunkLinearAttn

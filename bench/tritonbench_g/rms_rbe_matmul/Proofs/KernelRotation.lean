import bench.tritonbench_g.rms_rbe_matmul.Proofs.KernelStatistics

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem rotation_run (a : Operations α) (fp8 : Bool) (p : Parameters)
    (s t : State α) (values : TileIndex [16,64] → α) (h : TailReady p s t values) :
    ∃ u, run (engine a) (rotation fp8 p) t=some u ∧
      TailReady p s u (rotated a p s values) := by
  cases fp8 <;> cases hr : p.rotary
  all_goals
    simp [rotation,kernel,TypedSource.kernel,TypedSource.source,arguments,
      ComputeKernel.surfaceBody,run,step,evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,bop,natLt,numericLt,
      integralFloorDiv,integralMod,Region.cast,ofFloat,toFloat,State.setReg,
      engine,Algebra.withDefaultPrecision,resolvePrecision,h.product,h.tileM,h.tileN,
      hr,rotated,TileShape.product]
  all_goals first | exact h | constructor
  all_goals try simp [State.setReg,h.mem,h.pids,h.undef,h.batch,h.tileM,h.tileN,
    h.product,rotated,hr]
  all_goals funext i
  all_goals simp [-Fin.val_eq_zero_iff,rotated,rotatedPair,pairIndex,angle,row,pidM,pidN,hr,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,numeric,TileShape.product,
    TileShape.splitLastIndex,TileShape.appendAxisIndex,Nat.cast_add,Nat.cast_mul,
    Fin.ext_iff]

#axiomsClean rotation_run
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelSource

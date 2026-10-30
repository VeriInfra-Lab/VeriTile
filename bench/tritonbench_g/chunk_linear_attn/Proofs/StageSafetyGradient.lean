import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetySteps

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
open VeriTile.Bench.Repairs.ChunkGlaFwd.Geometry (packed_lt)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem gradient_entry_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) (capacity : GradientCapacity p bounds)
    (head : s.pids 2 < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds (gradientParts p scale).before s := by
  rw [← List.take_append_drop 9 (gradientParts p scale).before, Safety.traceSafeList_append]
  constructor
  · simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
    simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
      Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
    simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
      singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
    constructor
    · intro i j hk ht
      convert lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hk) capacity.query using 1 <;> ring
    · intro i j ht hk
      exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hk) capacity.key
  · intro t _
    apply pure_trace
    simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
    all_goals
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
      simp [Safety.safeOp_unfold]

theorem gradient_body_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (input : Memory α) (s : State α)
    (ctx : GradientReady p pid (p.B * p.H) input s)
    (fixed : GradientFixed M D p pid input scale s) (a : GradAcc p α) (ha : HasGrad p a s)
    (vb : Nat) (capacity : GradientCapacity p bounds)
    (head : pid.z < p.B * p.H) (keyBlock : pid.x < keyBlocks p) :
    Safety.traceSafeList (engine (withDot M D)) bounds (gradientParts p scale).body
      (s.setReg "i_v" .nat [] (fun _ => vb)) := by
  simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
    Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
    engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,transposeValues,
    ctx.ik,ctx.it,ctx.bh,ctx.nh,fixed.query,fixed.key,fixed.score,ha.query,ha.key,ha.score,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · intro i j ht hv
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hv) capacity.value
  · intro i j ht hv
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hv) capacity.incoming
  · intro i j hv hk
    convert lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head hk hv)
        (show p.B * p.H * (p.NT * p.K) * p.V ≤ bounds "history" by
          simpa only [Nat.mul_assoc] using capacity.history) using 1 <;> ring
  · intro i j hk hv
    simpa only [Nat.mul_assoc] using
      lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head hk hv)
        (show p.B * p.H * (p.NT * p.K) * p.V ≤ bounds "historyGradient" by
          simpa only [Nat.mul_assoc] using capacity.historyGradient)
  · intro i j ht hv
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ (packed_lt keyBlock head) ht hv)
      capacity.valuePartials

theorem gradient_finish_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (heads : Nat) (input : Memory α) (s : State α)
    (ctx : GradientReady p pid heads input s)
    (fixed : GradientFixed M D p pid input scale s) (a : GradAcc p α) (ha : HasGrad p a s)
    (capacity : GradientCapacity p bounds) (head : pid.z < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds (gradientParts p scale).after s := by
  simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [gradientParts,backwardQKV,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
    Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
    engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,transposeValues,FP.Structural.store,
    ctx.ik,ctx.it,ctx.bh,ctx.oi,fixed.query,fixed.key,fixed.score,ha.query,ha.key,ha.score,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  constructor
  · intro i j ht hk
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hk) capacity.queryGradient
  · intro i j ht hk
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hk) capacity.keyGradient

#axiomsClean gradient_entry_safe
#axiomsClean gradient_body_safe
#axiomsClean gradient_finish_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety

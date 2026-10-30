import bench.tritonbench_g.chunk_linear_attn.Proofs.StageSafetySteps

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem output_entry_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine (withDot M D)) bounds (outputParts p scale).before s := by
  apply pure_trace
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  all_goals
    simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
    simp [Safety.safeOp_unfold]

theorem output_body_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (input : Memory α) (s : State α)
    (ctx : OutputReady p pid input s) (a : OutputAcc p α) (ha : HasOutput p a s)
    (kb : Nat) (capacity : OutputCapacity p bounds)
    (head : pid.z < p.B * p.H) (time : pid.y < p.NT) :
    Safety.traceSafeList (engine (withDot M D)) bounds (outputParts p scale).body
      (s.setReg "i_k" .nat [] (fun _ => kb)) := by
  have ho := ha.1
  have hs := ha.2
  rw [← List.take_append_drop 6 (outputParts p scale).body, Safety.traceSafeList_append]
  constructor
  swap
  · intro t _
    apply pure_trace
    simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
    all_goals
      simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
      simp [Safety.safeOp_unfold]
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
    Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
    engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.iv,ctx.it,ctx.bh,ho,hs,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  refine ⟨?_, ?_, ?_⟩
  · intro i j ht hk
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hk) capacity.query
  · intro i j hk ht
    convert lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hk) capacity.key using 1 <;> ring
  · intro i j hk hv
    exact lt_of_lt_of_le (history_bound _ _ _ _ _ _ _ _ head time hk hv) capacity.history

theorem output_finish_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (bounds : RegionName → Nat) (pid : PID) (input : Memory α) (s : State α)
    (ctx : OutputReady p pid input s) (a : OutputAcc p α) (ha : HasOutput p a s)
    (capacity : OutputCapacity p bounds) (head : pid.z < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds (outputParts p scale).after s := by
  have ho := ha.1
  have hs := ha.2
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD]
  simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [outputParts,forwardO,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
    Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
    engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.iv,ctx.it,ctx.bh,ctx.mask,ho,hs,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  constructor
  · intro i j ht hv
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hv) capacity.value
  · intro i j ht hv
    exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hv) capacity.output

#axiomsClean output_entry_safe
#axiomsClean output_body_safe
#axiomsClean output_finish_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety

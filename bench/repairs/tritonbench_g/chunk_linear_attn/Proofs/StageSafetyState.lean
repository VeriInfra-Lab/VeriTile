import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.StageSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive withDot)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem state_entry_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (bounds : RegionName → Nat) (s : State α)
    (capacity : StateCapacity p rev seed final bounds) (head : s.pids 2 < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds
      (stateParts p scale rev seed final).before s := by
  cases rev <;> cases seed
  all_goals
    simp [stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Option.map,Option.getD,
      singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    simp [Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    intro i j hrow hcol
    simpa only [Nat.mul_assoc] using lt_of_lt_of_le
      (matrix_bound _ _ _ _ _ _ head hrow hcol) (capacity.initial rfl)

theorem state_finish_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (bounds : RegionName → Nat) (pid : PID)
    (input : Memory α) (s : State α) (ctx : StateReady pid input s)
    (capacity : StateCapacity p rev seed final bounds) (head : pid.z < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds
      (stateParts p scale rev seed final).after s := by
  cases rev <;> cases final
  all_goals
    simp [stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Option.map,Option.getD,
      singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,ctx.ik,ctx.iv,ctx.bh,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    simp [Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    intro i j hrow hcol
    simpa only [Nat.mul_assoc] using lt_of_lt_of_le
      (matrix_bound _ _ _ _ _ _ head hrow hcol) (capacity.final rfl)

theorem state_store_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (bounds : RegionName → Nat) (pid : PID)
    (input : Memory α) (s : State α) (ctx : StateReady pid input s)
    (n : Nat) (active : n < p.NT) (capacity : StateCapacity p rev seed final bounds)
    (head : pid.z < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds
      (stateStorePrefix p scale rev seed final)
      (s.setReg (loopName rev) .nat [] (fun _ => n)) := by
  have chunkBound := Geometry.chunk_lt p rev n active
  cases rev
  all_goals
    simp [stateStorePrefix,stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Option.map,Option.getD,
      singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,ctx.ik,ctx.iv,ctx.bh,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    simp [Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,evalOp_unfold,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    intro i j hrow hcol
    simpa only [chunk, Bool.false_eq_true, if_false, if_true, Nat.mul_assoc] using
      lt_of_lt_of_le (history_bound _ _ _ _ _ _ _ _ head chunkBound hrow hcol) capacity.history

theorem state_update_safe (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (scale : ℝ)
    (rev seed final : Bool) (bounds : RegionName → Nat) (pid : PID)
    (input : Memory α) (s : State α) (ctx : StateReady pid input s)
    (n : Nat) (pointers : StatePointers p pid rev n s)
    (capacity : StateCapacity p rev seed final bounds) (head : pid.z < p.B * p.H) :
    Safety.traceSafeList (engine (withDot M D)) bounds
      (stateUpdateSuffix p scale rev seed final) s := by
  have hl := pointers.left
  have hr := pointers.right
  cases rev
  all_goals
    simp only [Bool.false_eq_true, if_false, if_true] at hl hr
    simp [stateUpdateSuffix,stateParts,stateKernel,loopName,forwardH,backwardH,ComputeKernel.surfaceBody,
      Source.dynamicRange?,Option.map,Option.getD,
      singleton,Safety.traceSafeList_cons,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp_unfold,Safety.safeAddress_unfold,Safety.safeMask_unfold,
      Safety.addressSafe_unfold,Safety.maskActive_unfold,
      step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,
      engine,withDot,Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,hl,hr,
      BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  all_goals
    constructor
    · intro i j hk ht
      simpa only [Nat.mul_assoc, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using
        lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hk) capacity.left
    · intro i j ht hv
      exact lt_of_lt_of_le (matrix_bound _ _ _ _ _ _ head ht hv) capacity.right

#axiomsClean state_entry_safe
#axiomsClean state_finish_safe
#axiomsClean state_store_safe
#axiomsClean state_update_safe
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.StageSafety

import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyBasics

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPOutput
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

structure OutputCapacity (c : OutputGrid.Shape) (s : State α)
    (bounds : RegionName → Nat) : Prop where
  head : s.pids 2 < c.heads
  time : s.pids 1 < Geometry.ceilDiv c.T c.BT
  query : c.heads*c.T*c.K ≤ bounds "q"
  gate : c.heads*c.T*c.K ≤ bounds "g"
  history : c.heads*(Geometry.ceilDiv c.T c.BT*c.K)*c.V ≤ bounds "h"
  value : c.heads*c.T*c.V ≤ bounds "v"
  attention : c.heads*c.T*c.BT ≤ bounds "attention"
  output : c.heads*c.T*c.V ≤ bounds "o"

theorem output_entry (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (bounds : RegionName → Nat) (s : State α) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).before s := by
  apply pure_trace
  simp [parts,kernel,SourceBridge.outputNew,TypedOutput.original,TypedOutput.source,TypedOutput.fp32Original,
    OutputGrid.layout,OutputGrid.regions,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore]
  all_goals simp [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]

theorem output_body (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (bounds : RegionName → Nat) (initial current : State α) (f : Nat)
    (ctx : Context c initial current) (capacity : OutputCapacity c initial bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => f)) := by
  rw [←List.take_append_drop 8 (parts c scale).body,Safety.traceSafeList_append]
  constructor
  swap
  · intro t _
    simp [parts,kernel,SourceBridge.outputNew,TypedOutput.original,TypedOutput.source,TypedOutput.fp32Original,
      OutputGrid.layout,OutputGrid.regions,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore]
    simp only [singleton,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    simp [Safety.safeOp_unfold,singleton,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold]
    split <;> simp
  simp [parts,kernel,SourceBridge.outputNew,TypedOutput.original,TypedOutput.source,TypedOutput.fp32Original,
    OutputGrid.layout,OutputGrid.regions,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore]
  simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.iv,ctx.it,ctx.bh,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  refine ⟨?_,?_,?_⟩
  · intro i j row feature
    exact lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row feature) capacity.query
  · intro i j row feature
    exact lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row feature) capacity.gate
  · intro i j feature value
    have inner := Geometry.packed_lt capacity.time feature
    have backedAddress :=  FPReadBounds.tensor_bound capacity.head inner value
    convert lt_of_lt_of_le backedAddress capacity.history using 1 <;> ring

theorem output_finish (M : Algebra α) (D : DotPrimitive α) (c : OutputGrid.Shape) (scale : ℝ)
    (bounds : RegionName → Nat) (initial current : State α) (ctx : Context c initial current)
    (a : Index c → α) (ha : HasAcc c a current) (capacity : OutputCapacity c initial bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).after current := by
  simp only [HasAcc] at ha
  simp [parts,kernel,SourceBridge.outputNew,TypedOutput.original,TypedOutput.source,TypedOutput.fp32Original,
    OutputGrid.layout,OutputGrid.regions,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.prefixBefore]
  simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.iv,ctx.it,ctx.bh,ctx.causal,ha,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  refine ⟨?_,?_,?_⟩
  · intro i j row value
    exact lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row value) capacity.value
  · intro i j row
    convert lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row j.isLt) capacity.attention using 1 <;> ring
  · intro i j row value
    exact lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row value) capacity.output

#axiomsClean output_entry
#axiomsClean output_body
#axiomsClean output_finish
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety

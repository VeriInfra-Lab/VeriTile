import bench.repairs.tritonbench_g.chunk_gla_fwd.Proofs.NativeSafetyInterPrelude

namespace VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite FPInter
open FPOutput (DotPrimitive withDot)
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 8000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false

theorem inter_body (M : Algebra α) (D : DotPrimitive α)
    (c : ProducerGeometry.Shape) (scale : ℝ) (bounds : RegionName → Nat)
    (initial current : State α) (f : Nat) (ctx : Context c initial current)
    (capacity : InterCapacity c initial bounds) (enabled : enabled c initial) :
    Safety.traceSafeList (engine (withDot M D)) bounds (parts c scale).body
      (current.setReg "i_k" .nat [] (fun _ => f)) := by
  rw [←List.take_append_drop 13 (parts c scale).body,Safety.traceSafeList_append]
  constructor
  swap
  · intro t _
    apply pure_trace
    simp [parts,FPInter.inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,
      ProducerGeometry.layout,ProducerGeometry.regions,ComputeKernel.surfaceBody,
      Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
    all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
    all_goals simp [Safety.safeOp_unfold]
  simp [parts,FPInter.inner,outer,kernel,SourceBridge.interNew,TypedInter.original,TypedInter.source,
    ProducerGeometry.layout,ProducerGeometry.regions,ComputeKernel.surfaceBody,
    Source.conditional?,Source.dynamicRange?,Source.prefixBefore]
  simp only [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,safeBlockPtr2]
  simp [singleton,Safety.traceSafeList_cons,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,boundaryZero,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.it,ctx.ic,ctx.bh,ctx.ii,ctx.ij,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,TileShape.indexToList]
  refine ⟨?_,?_,?_,?_,?_⟩
  · intro i active
    exact lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head enabled.1 active) capacity.gate
  · intro i j row feature
    exact lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row feature) capacity.query
  · intro i j row feature
    exact lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row feature) capacity.gate
  · intro i j feature row
    convert lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row feature) capacity.key using 1 <;> ring
  · intro i j feature row
    convert lt_of_lt_of_le (FPReadBounds.tensor_bound capacity.head row feature) capacity.gate using 1 <;> ring

#axiomsClean inter_body
end
end VeriTile.Bench.Repairs.ChunkGlaFwd.NativeSafety

import bench.tritonbench_g.triton_attention.Proofs.AccessGeometry
import bench.tritonbench_g.triton_attention.Proofs.FPForwardPrologue

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
attribute [local simp 2000] AccessSafety.two_offset_pointer

/-- Actual source prologue: the only read is the unmasked query tile. -/
theorem forward_prologue_safe (rest : Algebra α) (P : Primitives α)
    (p : ForwardParameters) (s : State α) (bounds : RegionName→Nat)
    (query : ∀ i : FQuery p,forwardOffset p s i<bounds p.q) :
    Safety.traceSafeList (backend rest P) bounds (forwardPrologue p) s := by
  simp [forwardPrologue,forwardSource,forwardTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,backend,
    numeric,integralFloorDiv,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,
    BlockPtr.inBounds,BlockPtr.address_2d_offsets,TileShape.indexToList]
  intro i d
  simpa only [forwardOffset,forwardHeadRow,forwardQueryRow,Nat.add_assoc] using query (i,d,PUnit.unit)

#axiomsClean forward_prologue_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess

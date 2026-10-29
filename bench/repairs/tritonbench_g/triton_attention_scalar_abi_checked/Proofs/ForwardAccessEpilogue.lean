import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.AccessGeometry
import bench.repairs.tritonbench_g.triton_attention_scalar_abi_checked.Proofs.FPForwardComplete

namespace VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess
open VeriTile Triton FP.Structural Source
open ChunkRetention.Whole001
noncomputable section
variable {α : Type} [Inhabited α]
set_option maxHeartbeats 2000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem forward_epilogue_safe (rest : Algebra α) (P : Primitives α)
    (p : ForwardParameters) (initial current : State α) (block : Nat)
    (acc : ForwardAccumulator α p) (inv : ForwardInvariant P p initial block acc current)
    (bounds : RegionName→Nat)
    (denominator : ∀ i : FRow p,forwardRowOffset p initial i<bounds p.denominator)
    (maximum : ∀ i : FRow p,forwardRowOffset p initial i<bounds p.maximum)
    (output : ∀ i : FQuery p,forwardOffset p initial i<bounds p.output) :
    Safety.traceSafeList (backend rest P) bounds (forwardEpilogue p) current := by
  simp [forwardEpilogue,forwardSource,forwardTyped,ComputeKernel.surfaceBody,
    AccessSafety.singleton_safe,AccessSafety.pair_safe,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold,
    Safety.safeAddress_unfold,Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    step,store,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,Option.bind,backend,
    numeric,integralFloorDiv,ComputeDType.eraseDType,ofFloat,toFloat,bop,
    Broadcast.leftIndex,Broadcast.rightIndex,Region.cast,State.setReg,
    inv.start,inv.pids,inv.hz,inv.numerator,inv.denominator,inv.maximum,inv.outputPtr,
    forwardOutPointer,forwardQPointer,TileShape.indexToList,
    BlockPtr.inBounds_2d_offsets,BlockPtr.address_2d_offsets,
    State.scatter_regs,maskedScatter_regs]
  constructor
  · intro i
    exact denominator (i,PUnit.unit)
  constructor
  · intro i
    exact maximum (i,PUnit.unit)
  · intro i d _
    simpa only [forwardOffset,forwardHeadRow,forwardQueryRow,Nat.add_assoc] using output (i,d,PUnit.unit)

#axiomsClean forward_epilogue_safe
end
end VeriTile.Bench.Repairs.TritonAttentionScalarABI.SourceAccess

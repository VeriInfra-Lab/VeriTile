import bench.repairs.tritonbench_g.lightning_attention.KernelOnly.Proofs.SourceSafetyBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forwardEntry (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interForwardParts p).before s := by
  simp [interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg]
theorem reverseEntry (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interReverseParts p).before s := by
  simp [interReverseParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.qkOffset,ctx.vOffset,ctx.oOffset]

theorem deltaInit (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interFDeltaParts p).before s := by
  simp [interFDeltaParts,interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg]

theorem carryFinish (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry acc : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) (ha : InterFDelta p acc s)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interFDeltaParts p).after s := by
  simp only [InterFDelta] at ha
  simp [interFDeltaParts,interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,bop,State.setReg,ctx.total,ha]

theorem reverseDeltaInit (M : Algebra α) (D : DotPrimitive α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interRDeltaParts p).before s := by
  simp [interRDeltaParts,interRUpdateParts,interReverseParts,inter,ComputeKernel.surfaceBody,interForwardParts,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,State.setReg]

theorem reverseCarryFinish (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 : Nat) (input : Memory α) (carry acc : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 row2 input carry s) (ha : InterRDelta p acc s)
    (bounds : RegionName → Nat) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interRDeltaParts p).after s := by
  simp only [InterRDelta] at ha
  simp [interRDeltaParts,interRUpdateParts,interReverseParts,inter,ComputeKernel.surfaceBody,interForwardParts,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,numeric,bop,State.setReg,ctx.total,ha]

end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety

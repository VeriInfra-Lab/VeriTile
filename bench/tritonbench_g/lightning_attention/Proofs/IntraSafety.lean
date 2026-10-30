import bench.tritonbench_g.lightning_attention.Proofs.SourceSafetyBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem intra (M : Algebra α) (hm : MaskArithmetic M) (D : DotPrimitive α) (p : Parameters)
    (bounds : RegionName → Nat) (s : State α) (backed : HeadBounds p (s.pids 0) bounds) :
    TraceSafe (engine (withDot M D)) bounds (KernelOnly.intra p) s := by
  simp [TraceSafe,KernelOnly.intra,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr,Safety.safeCompute,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    hm.compare,numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex]

  exact ⟨fun t a h => backed.key _ _ h a.isLt,
    fun c t h => backed.value _ _ h c.isLt,
    fun t c h => backed.incoming _ _ h c.isLt,
    fun a t h => backed.query _ _ h a.isLt,
    fun t a h => backed.queryGradient _ _ h a.isLt,
    fun a t h => backed.keyGradient _ _ h a.isLt,
    fun t c h => backed.valueGradient _ _ h c.isLt⟩

#axiomsClean intra
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety

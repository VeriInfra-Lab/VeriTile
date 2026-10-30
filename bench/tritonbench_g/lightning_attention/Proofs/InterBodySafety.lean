import bench.tritonbench_g.lightning_attention.Proofs.SourceSafetyBounds
namespace VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaFwd.FPOutput (DotPrimitive)
noncomputable section
set_option maxHeartbeats 24000000
set_option maxRecDepth 30000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem queryBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s)
    (hi : s.regs .nat [] "i"=some (fun _ => block))
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interFQueryParts p).body
      (s.setReg "j" .nat [] (fun _ => sub)) := by
  have hg := ctx.inputs "incoming" (by simp)
  by_cases hb : 0<block
  all_goals
    simp [interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
    ctx.query,ctx.incoming,ctx.rows1,ctx.total,hi,hg,hb]

  exact ⟨fun t c h => backed.incoming _ _ h c.isLt,
    fun t a h => backed.queryGradient _ _ h a.isLt⟩

theorem deltaBody (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.E p.D)
    (s : State α) (ctx : InterFReady p head row1 row2 input carry s) (ha : InterFDelta p acc s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interFDeltaParts p).body
      (s.setReg "j" .nat [] (fun _ => sub)) := by
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  simp only [InterFDelta] at ha
  simp [interFDeltaParts,interFQueryParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.key,ctx.value,ctx.rows2,ha,hk,hv]

  exact ⟨fun c t h => backed.value _ _ h c.isLt,
    fun t a h => backed.key _ _ h a.isLt⟩

theorem reverseUpdate (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 block sub : Nat) (input : Memory α) (carry : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head (row1+p.CB) row2 input carry s)
    (hi : s.regs .int [] "i"=some (fun _ => (p.NT : Int)-1-block)) (hbnd : block<p.NT)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interRUpdateParts p).body
      (s.setReg "__rev_j" .int [] (fun _ => (sub : Int)+1-p.NC)) := by
  have hk := ctx.inputs "key" (by simp)
  have hv := ctx.inputs "value" (by simp)
  have hc : ((p.NT : Int)-1-block < (p.NT : Int)-1) ↔ block≠0 := by omega
  by_cases hb : block=0
  all_goals
    simp [interRUpdateParts,interReverseParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
      Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
      Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
      numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
      TileShape.axisDim,Region.cast,State.setReg,FP.Structural.store,
      ctx.key,ctx.value,ctx.dk,ctx.dv,ctx.rows1,ctx.total,hi,hk,hv,hc,hb,back_address,back_rows]

  exact ⟨fun t a h => backed.key _ _ h a.isLt,
    fun c t h => backed.value _ _ h c.isLt,
    fun a t h => backed.keyGradient _ _ h a.isLt,
    fun t c h => backed.valueGradient _ _ h c.isLt,
    fun a t h => backed.keyGradient _ _ h a.isLt,
    fun t c h => backed.valueGradient _ _ h c.isLt⟩

theorem reverseDelta (M : Algebra α) (D : DotPrimitive α) (p : Parameters)
    (head row1 row2 sub : Nat) (input : Memory α) (carry acc : Matrix α p.D p.E)
    (s : State α) (ctx : InterRReady p head row1 (row2+p.CB) input carry s) (ha : InterRDelta p acc s)
    (bounds : RegionName → Nat) (backed : HeadBounds p head bounds) :
    Safety.traceSafeList (engine (withDot M D)) bounds (interRDeltaParts p).body
      (s.setReg "__rev_j" .int [] (fun _ => (sub : Int)+1-p.NC)) := by
  have hq := ctx.inputs "query" (by simp)
  have hg := ctx.inputs "incoming" (by simp)
  simp only [InterRDelta] at ha
  simp [interRDeltaParts,interRUpdateParts,interReverseParts,interForwardParts,inter,ComputeKernel.surfaceBody,Source.dynamicRange?,Source.intRange?,Option.map,Option.getD,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeCompute_unfold,
    Safety.safeOp_unfold,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
    Tile.bop,Broadcast.leftIndex,Broadcast.rightIndex,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,Option.bind,engine,withDot,
    Algebra.withDefaultPrecision,resolvePrecision,ComputeDType.eraseDType,
    numeric,integralFloorDiv,integralMod,natLe,numericLe,numericLt,natLt,bop,ofFloat,toFloat,
    TileShape.axisDim,Region.cast,State.setReg,ctx.query,ctx.incoming,ctx.rows2,ha,hq,hg,
    back_address,back_rows]

  exact ⟨fun t c h => backed.incoming _ _ h c.isLt,
    fun a t h => backed.query _ _ h a.isLt⟩

#axiomsClean queryBody
#axiomsClean deltaBody
#axiomsClean reverseUpdate
#axiomsClean reverseDelta
end
end VeriTile.Bench.Repairs.LightningAttention.KernelOnly.SourceSafety

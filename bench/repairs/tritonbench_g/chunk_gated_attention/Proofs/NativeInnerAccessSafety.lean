import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeAccessSafety

namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
open NativeValue
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

structure InnerBounds (c : NativeValue.Shape) (base : Nat) (initial : State NativeGraph.Expr)
    (gateK hasSeed storeFinal : Bool) (bounds : RegionName → Nat) : Prop where
  seed : hasSeed=Bool.true → ∀ i : HIndex c, stateOffset c base initial i < bounds "h0"
  key : ∀ chunk, chunk<c.NT → ∀ i : KIndex c, keyOffset c base initial chunk i < bounds "k"
  value : ∀ chunk, chunk<c.NT → ∀ i : VIndex c, valueOffset c base initial chunk i < bounds "v"
  gateKey : gateK=Bool.true → ∀ chunk, chunk<c.NT → ∀ i : KIndex c,
    keyOffset c base initial chunk i < bounds "g"
  gateValue : gateK=Bool.false → ∀ chunk, chunk<c.NT → ∀ i : VIndex c,
    valueOffset c base initial chunk i < bounds "g"
  keyEnd : gateK=Bool.true → ∀ chunk, chunk<c.NT → ∀ i : Fin c.BK,
    keyEndpoint c base initial chunk i < bounds "g"
  valueEnd : gateK=Bool.false → ∀ chunk, chunk<c.NT → ∀ i : Fin c.BV,
    valueEndpoint c base initial chunk i < bounds "g"
  history : ∀ chunk, chunk<c.NT → ∀ i : HIndex c, historyOffset c base initial chunk i < bounds "h"
  final : storeFinal=Bool.true → ∀ i : HIndex c, stateOffset c base initial i < bounds "ht"

theorem inner_entry_safe (c : NativeValue.Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (s : State NativeGraph.Expr) (bounds : RegionName → Nat)
    (hb : InnerBounds c base s gateK hasSeed storeFinal bounds) :
    Safety.traceSafeList (engine algebra) bounds (parts c base gateK hasSeed storeFinal).before s := by
  cases hasSeed
  all_goals simp [parts,NativeValue.source,NativeSource.hOriginal,NativeSource.hSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Safety.traceSafeList,Safety.stepSafe,
    Safety.safeExpr,Safety.safeCompute,Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim,
    engine,Algebra.withDefaultPrecision,resolvePrecision,algebra,ScalarGraph.algebra,NativeGraph.algebra]
  exact fun i j => hb.seed rfl (i,j,PUnit.unit)

theorem inner_body_safe (c : NativeValue.Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (chunk : Nat) (acc : HIndex c → NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc)
    (bounds : RegionName → Nat) (hb : InnerBounds c base initial gateK hasSeed storeFinal bounds)
    (hchunk : chunk<c.NT) :
    Safety.traceSafeList (engine algebra) bounds (parts c base gateK hasSeed storeFinal).body
      (current.setReg "i_t" .nat [] (fun _=>chunk)) := by
  have hacc := ha
  unfold HasAcc at hacc
  cases gateK
  all_goals simp [parts,NativeValue.source,NativeSource.hOriginal,NativeSource.hSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Safety.traceSafeList,Safety.stepSafe,
    Safety.safeExpr,Safety.safeCompute,Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim,
    ctx.bh,ctx.ik,ctx.iv,hacc,FP.Structural.store,
    engine,Algebra.withDefaultPrecision,resolvePrecision,algebra,ScalarGraph.algebra,NativeGraph.algebra]
  · exact ⟨fun i j => hb.history chunk hchunk (i,j,PUnit.unit),
      fun i j => hb.key chunk hchunk (i,j,PUnit.unit),
      fun i j => hb.value chunk hchunk (i,j,PUnit.unit),
      fun i u _ => by simpa [TileShape.indexToList,valueEndpoint] using hb.valueEnd rfl chunk hchunk i,
      fun i j => hb.gateValue rfl chunk hchunk (i,j,PUnit.unit)⟩
  · exact ⟨fun i j => hb.history chunk hchunk (i,j,PUnit.unit),
      fun i j => hb.key chunk hchunk (i,j,PUnit.unit),
      fun i j => hb.value chunk hchunk (i,j,PUnit.unit),
      fun i u _ => by simpa [TileShape.indexToList,keyEndpoint] using hb.keyEnd rfl chunk hchunk i,
      fun i j => hb.gateKey rfl chunk hchunk (i,j,PUnit.unit)⟩

theorem inner_finish_safe (c : NativeValue.Shape) (base : Nat) (gateK hasSeed storeFinal : Bool)
    (initial current : State NativeGraph.Expr) (acc : HIndex c → NativeGraph.Expr)
    (ctx : Context c base initial current) (ha : HasAcc c current acc)
    (bounds : RegionName → Nat) (hb : InnerBounds c base initial gateK hasSeed storeFinal bounds) :
    Safety.traceSafeList (engine algebra) bounds (parts c base gateK hasSeed storeFinal).after current := by
  have hacc := ha
  unfold HasAcc at hacc
  cases storeFinal
  all_goals simp [parts,NativeValue.source,NativeSource.hOriginal,NativeSource.hSource,
    ComputeKernel.surfaceBody,Source.dynamicRange?,Safety.traceSafeList,Safety.stepSafe,
    Safety.safeExpr,Safety.safeCompute,Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,
    Safety.addressSafe,Safety.maskActive,run,step,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
    Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
    natLt,natLe,numericLt,numericLe,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim,
    ctx.bh,ctx.ik,ctx.iv,hacc,FP.Structural.store,
    engine,Algebra.withDefaultPrecision,resolvePrecision,algebra,ScalarGraph.algebra,NativeGraph.algebra]
  exact fun i j => hb.final rfl (i,j,PUnit.unit)

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSafety

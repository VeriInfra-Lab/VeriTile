import bench.repairs.tritonbench_g.chunk_gated_attention.Proofs.NativeGrid
import VeriTile.Triton.Float.StructuralSafety

/-! Actual evaluated Structural accesses of both native sources. Bounds name
source addresses only; no numerical or successful-execution premise occurs. -/
namespace VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSafety
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open VeriTile.Bench.Repairs.ChunkGlaSimple
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

structure PrefixBounds (c : NativePrefix.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (bounds : RegionName → Nat) : Prop where
  input : NativePrefix.enabled c base s → ∀ i : NativePrefix.Index c,
    NativePrefix.offset c base s i < bounds "s"
  output : NativePrefix.enabled c base s → ∀ i : NativePrefix.Index c,
    NativePrefix.offset c base s i < bounds "o"

theorem prefix_safe (c : NativePrefix.Shape) (base : Nat) (s : State NativeGraph.Expr)
    (bounds : RegionName → Nat) (hb : PrefixBounds c base s bounds) :
    TraceSafe (engine NativeValue.algebra) bounds (NativePrefix.source c base) s := by
  by_cases active : NativePrefix.enabled c base s
  all_goals
    have condition := active
    unfold NativePrefix.enabled NativePrefix.linear at condition
    simp [TraceSafe,NativePrefix.source,NativeSource.cumOriginal,ComputeKernel.surfaceBody,
      Safety.traceSafeList,Safety.stepSafe,Safety.safeExpr,Safety.safeCompute,
      Safety.safeOp.eq_def,Safety.safeAddress,Safety.safeMask,Safety.addressSafe,Safety.maskActive,
      run,step,FP.Structural.store,FP.Structural.evalExpr,evalComputeOp,evalOp_unfold,
      Option.bind,ComputeDType.eraseDType,numeric,integralFloorDiv,integralMod,bop,
      natLt,natLe,numericLt,numericLe,ofFloat,toFloat,Region.cast,State.setReg,TileShape.axisDim,
      engine,Algebra.withDefaultPrecision,resolvePrecision,NativeValue.algebra,ScalarGraph.algebra,NativeGraph.algebra,condition]
  exact ⟨fun i j => hb.input active (i,j,PUnit.unit),fun i j => hb.output active (i,j,PUnit.unit)⟩

end
end VeriTile.Bench.Repairs.ChunkGatedAttention.NativeSafety

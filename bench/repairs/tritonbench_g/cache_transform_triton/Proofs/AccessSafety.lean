import bench.repairs.tritonbench_g.cache_transform_triton.Proofs.Execution
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.CacheTransform
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The unconditional metadata load and both masked byte accesses follow the
actual source evaluation. No output formula or abstract safety premise is used. -/
theorem access_safe {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters)
    (s : State α) (bounds : RegionName → Nat)
    (rows : ∀ i : TileIndex [p.block],safe p s i/p.rowBytes<bounds p.rows)
    (payload : ∀ i : TileIndex [p.block],index p s i<p.count → inputOffset p s i<bounds p.cache)
    (output : ∀ i : TileIndex [p.block],index p s i<p.count → safe p s i<bounds p.output) :
    TraceSafe M bounds (source p) s := by
  simp [TraceSafe,Safety.traceSafeList,Safety.stepSafe,Safety.safeAlg,Safety.safeExpr,
    Safety.safeCompute,Safety.safeOp_unfold,Safety.safeMask,Safety.safeAddress,Safety.addressSafe,
    Safety.maskActive,source,body,ComputeKernel.surfaceBody,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,
    FP.Structural.evalOffsets,FP.Structural.store,FP.Structural.numeric,
    FP.Structural.bop,FP.Structural.natLt,ComputeDType.eraseDType,Region.cast,Option.bind,
    Broadcast.leftIndex,Broadcast.rightIndex,integralFloorDiv,integralMod,IntegerExpr.sameShape]
  simpa [safe,inputOffset,index_eq] using And.intro rows (And.intro payload output)

end
end VeriTile.Bench.Repairs.CacheTransform

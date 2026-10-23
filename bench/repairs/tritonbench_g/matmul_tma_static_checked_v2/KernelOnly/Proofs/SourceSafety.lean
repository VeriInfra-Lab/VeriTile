import bench.repairs.tritonbench_g.matmul_tma_static_checked_v2.KernelOnly.Kernels
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceSafety
open VeriTile.Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 5000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The actual unmasked block-pointer loads and store require every tile
address. Forming the block pointers itself does not read their backing. -/
theorem source_safe {α : Type} [Inhabited α] (M : Algebra α) (D : Dot α)
    (p : Parameters) (half : Bool) (s : State α) (bounds : RegionName → Nat)
    (ha : ∀ i : AIndex p, aOffset p i < bounds p.a)
    (hb : ∀ i : BIndex p, bOffset p i < bounds p.b)
    (hc : ∀ i : CIndex p, cOffset p i < bounds p.c) :
    TraceSafe (backend M D) bounds (source p half) s := by
  cases half
  all_goals
    simp [TraceSafe, Safety.traceSafeList_unfold, Safety.stepSafe_unfold,
      Safety.safeExpr_unfold, Safety.safeCompute_unfold, Safety.safeOp_unfold,
      Safety.safeAddress_unfold, Safety.safeMask_unfold, Safety.addressSafe_unfold,
      Safety.maskActive_unfold, Safety.safeAlg_unfold, source, typed,
      FP.Structural.exec, ComputeKernel.surfaceBody, run, step,
      evalExpr, evalComputeOp, evalOp_unfold, store, Option.bind, ComputeDType.eraseDType,
      numeric, bop, Region.cast, BlockPtr.inBounds, BlockPtr.address_2d_zero_offsets,
      TileShape.indexToList, backend, outputValues, aValues, bValues,
      aOffset, bOffset, cOffset, outputDType, ofFloat, toFloat]
  all_goals exact ⟨fun i k => ha (i,k,PUnit.unit), fun k j => hb (k,j,PUnit.unit),
    fun i j => hc (i,j,PUnit.unit)⟩

#axiomsClean source_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.MatmulTma.StaticCheckedV2.SourceSafety

import bench.repairs.tritonbench_g.destindex_copy_kv2.Proofs.Tile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.DestindexCopyKv2.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- The actual kv2 source has an unconditional signed metadata load and only
a head mask. The explicit width premise of the public IO theorem supplies
backing for every feature lane; it is not silently inserted into the source. -/
theorem source_trace_safe {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Layout) (s : State α) (bounds : RegionName → Nat)
    (metadata : s.pids 0 < bounds p.Dest_loc.cast)
    (input : ∀ i, active p i → inputOffset p s i < bounds p.K)
    (output : ∀ i, active p i → outputOffset p s i < bounds p.Out) :
    TraceSafe (rawBackend rest) bounds (kernel p) s := by
  simp [TraceSafe,kernel,
    VeriTile.Bench.TritonBenchG.DestindexCopyKv2.TypedSource.kernel,ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    copyBackend,rawBackend,natLt,ComputeDType.eraseDType,numeric,bop,ofFloat,toFloat,
    Region.cast,store,TileShape.dropInsertedIndex]
  exact ⟨metadata, fun h d hh => input (h,d,PUnit.unit) hh,
    fun h d hh => output (h,d,PUnit.unit) hh⟩

#axiomsClean source_trace_safe
end
end VeriTile.Bench.Repairs.DestindexCopyKv2.Tile

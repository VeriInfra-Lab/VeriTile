import bench.tritonbench_g.destindex_copy_kv1.Proofs.Tile
import VeriTile.Triton.Float.StructuralSafety

namespace VeriTile.Bench.Repairs.DestindexCopyKv1.Tile
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Actual source access safety includes the unconditional scalar destination
load as well as both masked payload dimensions. Output addresses retain the
source's signed integer expression and conversion to the regional offset. -/
theorem source_trace_safe {α : Type} [Inhabited α] (rest : Algebra α)
    (p : Layout) (s : State α) (bounds : RegionName → Nat)
    (metadata : s.pids 0<bounds p.Dest_loc.cast)
    (input : ∀ i,active p i → inputOffset p s i<bounds p.K)
    (output : ∀ i,active p i → outputOffset p s i<bounds p.Out) :
    TraceSafe (rawBackend rest) bounds (kernel p) s := by
  simp [TraceSafe,kernel,
    ComputeKernel.surfaceBody,
    Safety.traceSafeList_unfold,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeCompute_unfold,Safety.safeOp_unfold,Safety.safeAddress_unfold,
    Safety.safeMask_unfold,Safety.addressSafe_unfold,Safety.maskActive_unfold,
    run,step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,evalDeltas,Option.bind,
    copyBackend,rawBackend,natLt,ComputeDType.eraseDType,numeric,bop,ofFloat,toFloat,
    Region.cast,store,TileShape.dropInsertedIndex]
  constructor
  · exact metadata
  constructor
  · intro h d hh hd
    exact input (h,d,PUnit.unit) ⟨hh,hd⟩
  · intro h d hh hd
    exact output (h,d,PUnit.unit) ⟨hh,hd⟩

#axiomsClean source_trace_safe
end
end VeriTile.Bench.Repairs.DestindexCopyKv1.Tile

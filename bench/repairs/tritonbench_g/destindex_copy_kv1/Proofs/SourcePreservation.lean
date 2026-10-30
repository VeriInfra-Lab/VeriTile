import bench.repairs.tritonbench_g.destindex_copy_kv1.Proofs.Tile

/-! The old Lean inserted an extra same-FP32 store cast. The actual Python
has no such statement. Complete execution agrees in the existing raw-copy
interpretation, which already makes homogeneous FP32 copies identity on words.
No cast identity is assumed for arbitrary floating algebras. -/
namespace VeriTile.Bench.Repairs.DestindexCopyKv1.SourcePreservation
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (copyBackend)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

theorem original {A : Type} [Inhabited A] (rest : Algebra A) (p : Layout) (s : State A) :
    FP.Structural.exec (Tile.rawBackend rest) (kernel p) s =
      FP.Structural.exec (Tile.rawBackend rest) (Legacy.kernel p) s := by
  simp [kernel, Legacy.kernel,
    Layout.source,
    VeriTile.Bench.TritonBenchG.DestindexCopyKv1.TypedSource.kernel,
    ComputeKernel.surfaceBody, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, Option.bind, copyBackend, Tile.rawBackend,
    natLt, ComputeDType.eraseDType, numeric, bop, ofFloat, toFloat, Region.cast,
    FP.Structural.store, TileShape.dropInsertedIndex]

#axiomsClean original

theorem candidate {A : Type} [Inhabited A] (rest : Algebra A) (p : Layout) (s : State A) :
    FP.Structural.exec (Tile.rawBackend rest) (indexCommutedKernel p) s =
      FP.Structural.exec (Tile.rawBackend rest) (Legacy.indexCommutedKernel p) s := by
  simp [indexCommutedKernel, Legacy.indexCommutedKernel,
    Layout.source,
    VeriTile.Bench.TritonBenchG.DestindexCopyKv1.TypedSource.kernel,
    ComputeKernel.surfaceBody, FP.Structural.exec, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, evalOffsets, evalDeltas, Option.bind, copyBackend, Tile.rawBackend,
    natLt, ComputeDType.eraseDType, numeric, bop, ofFloat, toFloat, Region.cast,
    FP.Structural.store, TileShape.dropInsertedIndex]

#axiomsClean candidate

end
end VeriTile.Bench.Repairs.DestindexCopyKv1.SourcePreservation

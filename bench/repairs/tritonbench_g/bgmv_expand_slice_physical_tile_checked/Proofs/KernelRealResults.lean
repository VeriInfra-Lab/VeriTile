import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.KernelRaw
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Real

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.KernelRealResults
open VeriTile Triton FP.Structural Direct FPValues
noncomputable section

/-- Complete signed shared-source grid, independent initial-array GEMV with
optional output addition, and exact raw tagged-memory frame. -/
theorem correct (h : Call) (data : Data ℝ)
    (valid : ∀ b, b < h.batches → data.indices b = -1 ∨ 0 ≤ data.indices b) :
    ∃ final, Signed.Grid.run Real.algebra Real.reduction h (Geometry.programs h) data = some final ∧
      (∀ p : Geometry.Program h, data.indices p.batch ≠ -1 →
        final.output (SourceGrid.address h p) = Real.formula h p data) ∧
      (∀ r off, KernelRaw.Outside h data r off → final.cells r off = data.cells r off) := by
  obtain ⟨final,run,values,frame⟩ := KernelRaw.complete Real.algebra Real.reduction h data valid
  exact ⟨final,run,fun p active => (values p active).trans (Real.value_formula h p data),frame⟩

#axiomsClean correct
#specNonCircular Real.formula avoiding [Signed.Grid.run,FP.Structural.exec,SourceGrid.fold,Surface.source]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.KernelRealResults

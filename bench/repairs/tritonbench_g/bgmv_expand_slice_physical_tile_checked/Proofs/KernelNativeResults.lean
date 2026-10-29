import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Proofs.KernelRaw
import bench.repairs.tritonbench_g.bgmv_expand_slice_physical_tile_checked.Native

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.KernelNativeResults
open VeriTile Triton FP.Structural Direct FPValues
noncomputable section

/-- The actual signed source executes over raw instruction-expression cells.
Only the same five fixed-arity native instruction laws relate actual/selected
observations; all active outputs and the full tagged-cell frame are retained. -/
theorem fp_equiv (h : Call) (data : Data Native.Expr) (plans : Native.Plans)
    (valid : ∀ b, b < h.batches → data.indices b = -1 ∨ 0 ≤ data.indices b)
    (actual selected : Native.Instructions) (numeric : Native.LocalConformance actual selected) :
    ∃ final, Signed.Grid.run Native.algebra (Native.reduction plans) h (Geometry.programs h) data = some final ∧
      (∀ p : Geometry.Program h, data.indices p.batch ≠ -1 →
        (final.output (FPGrid.address h p)).eval actual =
          (FPGrid.value Native.algebra (Native.reduction plans) h p data).eval selected) ∧
      (∀ r off, KernelRaw.Outside h data r off → final.cells r off = data.cells r off) := by
  obtain ⟨final,run,values,frame⟩ := KernelRaw.complete Native.algebra (Native.reduction plans) h data valid
  refine ⟨final,run,?_,frame⟩
  intro p active
  rw [values p active,Native.instructions_eq actual selected numeric]

#axiomsClean fp_equiv
#specNonCircular FPGrid.value avoiding [Signed.Grid.run,FP.Structural.exec,FPGrid.fold,Surface.source]
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.KernelNativeResults

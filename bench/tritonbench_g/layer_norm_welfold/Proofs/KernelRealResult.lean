import bench.tritonbench_g.layer_norm_welfold.Proofs.ReadableGrid
import bench.tritonbench_g.layer_norm_welfold.Proofs.Mathematics
import bench.tritonbench_g.layer_norm_welfold.Proofs.KernelBoundary
import bench.tritonbench_g.layer_norm_welfold.Proofs.NativeTree

namespace VeriTile.Bench.Repairs.LayerNormWelfold.KernelRealResult
open VeriTile.Triton FP.Structural Reuse Samples
noncomputable section

/-- Independent mathematics for the complete source grid over raw memory. -/
theorem kernel_real (rest : Algebra ℝ) (s : State ℝ) (xv wv bv : Nat → ℝ)
    (S D : Nat) (Y Mean Rstd : RegionName) (c : Configuration) (treeComplete : c.tree.Complete)
    (sep : Separation "X" "W" "Bias" Y Mean Rstd)
    (hx : ∀ row, row < S → ∀ col, col < D → (s.mem "X" (col + D * row)).read .real = xv (col + D * row))
    (hw : ∀ col, col < D → (s.mem "W" col).read .real = wv col)
    (hb : ∀ col, col < D → (s.mem "Bias" col).read .real = bv col) :
    ∃ final,
      SourceCode.launch (withEquality (ideal rest) (fun _ _ => Bool.false)) Bool.false s Y Mean Rstd S D c.block
        (SourceBinding.tree c.tree) = some final ∧
      (∀ row, row < S → final.mem Mean row = .mk .real (Math.mean xv D row)) ∧
      (∀ row, row < S → final.mem Rstd row = .mk .real (Math.inverse xv D row)) ∧
      (∀ row, row < S → ∀ col, col < D → final.mem Y (col + D * row) =
        .mk .bf16 (Math.output xv wv bv D row col)) ∧
      (∀ q off, gridMiss Y Mean Rstd S D q off → final.mem q off = s.mem q off) := by
  obtain ⟨final, execution, effect⟩ := Readable.whole_grid (ideal rest) (fun _ _ => Bool.false)
    s "X" "W" "Bias" Y Mean Rstd S D c.block c.tree Bool.false c.positive sep (List.range S)
  have hm : ∀ row, row < S → meanValue (ideal rest) (atRow s row) "X" D c.block c.tree =
      Math.mean (xv) D row := by
    intro row hr
    rw [mean_real rest _ _ _ _ _ c.positive treeComplete]
    exact KernelBoundary.mean_logical s _ S D row hr hx
  have hi : ∀ row, row < S → inverseValue (ideal rest) (atRow s row) "X" D c.block c.tree =
      Math.inverse (xv) D row := by
    intro row hr
    rw [inverse_real rest _ _ _ _ _ c.positive treeComplete]
    exact KernelBoundary.inverse_logical s _ S D row hr hx
  refine ⟨final, execution, ?_, ?_, ?_, ?_⟩
  · intro row hr
    simpa only [hm row hr] using effect.mean row (List.mem_range.mpr hr)
  · intro row hr
    simpa only [hi row hr] using effect.rstd row (List.mem_range.mpr hr)
  · intro row hr col hc
    have output := effect.output row (List.mem_range.mpr hr) col hc
    simp only [rowOutput] at output
    rw [hm row hr, hi row hr] at output
    simp only [normalizeValue, ideal] at output
    have hxr : (s.mem "X" (col + D * row)).read .real = xv (col + D * row) :=
      hx row hr col hc
    have hwr : (s.mem "W" col).read .real = wv col := hw col hc
    have hbr : (s.mem "Bias" col).read .real = bv col := hb col hc
    rw [hxr, hwr, hbr] at output
    exact output
  · intro q off miss
    exact effect.frame q off (fun row hr => miss row (List.mem_range.mp hr))

#specNonCircular Math.mean avoiding [SourceCode.rowKernel, SourceCode.launch]
#specNonCircular Math.variance avoiding [SourceCode.rowKernel, SourceCode.launch]
#specNonCircular Math.output avoiding [SourceCode.rowKernel, SourceCode.launch]
#axiomsClean kernel_real
end
end VeriTile.Bench.Repairs.LayerNormWelfold.KernelRealResult

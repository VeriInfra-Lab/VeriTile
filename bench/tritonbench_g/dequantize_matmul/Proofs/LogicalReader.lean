import bench.tritonbench_g.dequantize_matmul.IO
import bench.tritonbench_g.dequantize_matmul.Proofs.IOGrid

namespace VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO
open VeriTile.Triton FP.Structural KernelSource KernelModels
noncomputable section

theorem signed_pullback (p : Parameters) :
    (fun raw => LogicalSignedBytes (inputEquiv ℝ p raw)) = SignedBytes := by
  funext raw
  apply propext
  change (∀ row col, ∃ word : BitVec 8,
    quantized raw (finProdFinEquiv (row,col)) = word.toInt) ↔
      (∀ i, ∃ word : BitVec 8, quantized raw i = word.toInt)
  constructor
  · intro h i
    have hi := h (finProdFinEquiv.symm i).1 (finProdFinEquiv.symm i).2
    change (∃ word : BitVec 8, quantized raw
      (finProdFinEquiv (finProdFinEquiv.symm i)) = word.toInt) at hi
    simpa only [Equiv.apply_symm_apply] using hi
  · intro h row col
    exact h (finProdFinEquiv (row,col))

theorem product_transport (p : Parameters) :
    (fun raw => (outputEquiv ℝ p).symm (fun row col =>
      ((inputEquiv ℝ p raw).weights row col : ℝ) * (inputEquiv ℝ p raw).scales col)) =
      result (fun q scale => (q : ℝ) * scale) := by
  funext raw o j
  fin_cases o
  rcases j with ⟨i,u⟩
  cases u
  change (quantized raw (finProdFinEquiv (finProdFinEquiv.symm i)) : ℝ) *
    scale raw (finProdFinEquiv.symm i).2 =
      (quantized raw i : ℝ) * scale raw (column p i)
  rw [Equiv.apply_symm_apply]
  rfl

/-- Bijective coordinates preserve the full complete-grid judgment and domain. -/
theorem logical_iff (rest : Algebra ℝ) (p : Parameters) :
    (logicalIO rest p).ImplementsOn LogicalSignedBytes
      (fun xs row col => (xs.weights row col : ℝ) * xs.scales col) ↔
    (sourceIO (realAlgebra rest) p Bool.false).ImplementsOn SignedBytes
      (result (fun q scale => (q : ℝ) * scale)) := by
  change (sourceIO (realAlgebra rest) p Bool.false).ImplementsOn
    (fun raw => LogicalSignedBytes (inputEquiv ℝ p raw))
    (fun raw => (outputEquiv ℝ p).symm (fun row col =>
      ((inputEquiv ℝ p raw).weights row col : ℝ) * (inputEquiv ℝ p raw).scales col)) ↔ _
  exact Iff.of_eq (congrArg₂ (fun domain formula =>
    (sourceIO (realAlgebra rest) p Bool.false).ImplementsOn domain formula)
      (signed_pullback p) (product_transport p))

theorem logical_correct (rest : Algebra ℝ) (p : Parameters)
    (hK : 0 < p.BK) (hN : 0 < p.BN) (sep : KernelGrid.Separate p) :
    (logicalIO rest p).ImplementsOn LogicalSignedBytes
      (fun xs row col => (xs.weights row col : ℝ) * xs.scales col) := by
  apply (logical_iff rest p).mpr
  intro values _
  exact implements (realAlgebra rest) p hK hN sep (fun q scale => (q : ℝ) * scale)
    (fun _ _ => rfl) values trivial

end
end VeriTile.Bench.Repairs.DequantizeMatmul.PublicIO

import bench.tritonbench_g.softmax_triton2.IO
import bench.tritonbench_g.softmax_triton2.Proofs.Binding
import bench.tritonbench_g.softmax_triton1.Proofs.KernelMath

namespace VeriTile.Bench.Repairs.SoftmaxTriton2.RealProof
open VeriTile.Triton
open VeriTile.Bench.Repairs.SoftmaxTriton1 Contiguous
open scoped VeriTile.Triton.MaskedKernelIO₁
noncomputable section

/-- Successful mathematical execution, access bounds, each active output,
and preservation of every exterior flat-memory cell. -/
theorem io_correct (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) :
    Spec.Real (softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) := by
  refine MaskedKernelIO₁.Implements.intro _ (row_flattenOk ..) ?_ ?_
  · intro bounds s hin hout _
    exact row_safe out inp ib ob n 1 n 1 n B hB bounds s
      (by simpa [softmaxIO] using hin) (by simpa [softmaxIO] using hout)
  · intro s xs input
    obtain ⟨t,he⟩ := row_exec out inp ib ob n 1 n 1 n B hB s
    refine ⟨t,he,?_,?_⟩
    · intro i hi
      have value := row_value out inp ib ob n 1 n 1 n B hB
        (by intro i j _ _ eq; apply Fin.ext; simpa using eq) s t he i hi
      have same : ∀ j : Fin B, j.val < n →
          s.readMem inp (ib+s.pid*n+j.val*1) = xs j := by
        simpa [softmaxIO] using input
      rw [VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_congr n B _ xs same i,
        VeriTile.Bench.TritonBenchG.SoftmaxTriton1.softmaxSpec_eq_formula n B hB xs i hi] at value
      simpa [softmaxIO] using value
    · intro r a miss _
      apply row_frame out inp ib ob n 1 n 1 n B hB s t he r a
      intro i hi eq
      rcases miss with hr | ha
      · exact hr eq.1.symm
      · exact ha i hi (by simpa [softmaxIO] using eq.2.symm)

/-- A covering tile's masked sum is exactly the sum over all logical columns.
This is an equality in Real arithmetic, with no numerical assumption. -/
theorem masked_sum_eq_logical (n B : Nat) (cover : n ≤ B) (xs : Fin B → ℝ) :
    (∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) =
      ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j)) := by
  classical
  let f : Nat → ℝ := fun j => if hj : j < B then Real.exp (xs ⟨j, hj⟩) else 0
  calc
    _ = ∑ j ∈ Finset.range n, f j := by
      convert RealModel.padded_sum_eq n B cover f using 1
      apply Finset.sum_congr rfl
      intro j _
      simp only [f, dif_pos j.isLt]
    _ = _ := by
      rw [← Fin.sum_univ_eq_sum_range]
      apply Finset.sum_congr rfl
      intro j _
      simp only [f, dif_pos (lt_of_lt_of_le j.isLt cover)]
      rfl

/-- The displayed finite-column formula agrees with Contract.reference.
The extension outside the input tile is irrelevant because `n ≤ B`. -/
theorem logical_formula_eq_reference (n B : Nat) (cover : n ≤ B)
    (xs : Fin B → ℝ) (i : Fin B) :
    Real.exp (xs i) / (∑ j : Fin n, Real.exp (xs (Fin.castLE cover j))) =
      reference n (fun k => if hk : k < B then xs ⟨k, hk⟩ else 0) i.val := by
  unfold reference
  rw [← Fin.sum_univ_eq_sum_range]
  simp only [dif_pos i.isLt]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  simp only [dif_pos (lt_of_lt_of_le j.isLt cover)]
  rfl

/-- Full logical-row correctness follows from the preserved window theorem
and the explicit covering-tile condition. -/
theorem logical_io_correct (out inp : RegionName) (ib ob n B : Nat)
    (hB : 0 < B) (cover : n ≤ B) :
    Spec.Real (softmaxIO out inp ib ob n B ⊨ fun xs i =>
      Real.exp (xs i) / ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j))) := by
  simpa only [masked_sum_eq_logical n B cover] using io_correct out inp ib ob n B hB


end
end VeriTile.Bench.Repairs.SoftmaxTriton2.RealProof

import bench.tritonbench_g.softmax_triton1.Proofs.Real

namespace VeriTile.Bench.Repairs.SoftmaxTriton1.RealModel
open VeriTile.Triton
noncomputable section

/-- Padding contributes no term to the independent row sum. -/
theorem padded_sum_eq (n B : Nat) (cover : n≤B) (f : Nat → ℝ) :
    (∑ j : Fin B, if j.val<n then f j.val else 0)=∑ j ∈ Finset.range n, f j := by
  rw [Fin.sum_univ_eq_sum_range (fun j => if j<n then f j else 0)]
  calc
    (∑ j ∈ Finset.range B, if j<n then f j else 0)=
        ∑ j ∈ Finset.range n, if j<n then f j else 0 := by
      symm
      exact Finset.sum_subset (Finset.range_mono cover) (by
        intro j _ outside
        simp only [Finset.mem_range] at outside
        simp [outside])
    _=∑ j ∈ Finset.range n, f j := Finset.sum_congr rfl (by
      intro j hj
      exact if_pos (Finset.mem_range.mp hj))

/-- Reindex the active tile terms by exactly the logical columns. -/
theorem masked_sum_eq_logical (n B : Nat) (cover : n ≤ B) (xs : Fin B → ℝ) :
    (∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) =
      ∑ j : Fin n, Real.exp (xs (Fin.castLE cover j)) := by
  classical
  let f : Nat → ℝ := fun j => if hj : j < B then Real.exp (xs ⟨j, hj⟩) else 0
  calc
    _ = ∑ j ∈ Finset.range n, f j := by
      convert padded_sum_eq n B cover f using 1
      apply Finset.sum_congr rfl
      intro j _
      simp only [f, dif_pos j.isLt]
    _ = _ := by
      rw [← Fin.sum_univ_eq_sum_range]
      apply Finset.sum_congr rfl
      intro j _
      simp only [f, dif_pos (lt_of_lt_of_le j.isLt cover)]
      rfl

/-- Independent mathematical softmax; B does not occur in its formula. -/
def formula (n _B : Nat) (xs : Nat → ℝ) (i : Nat) : ℝ :=
  Real.exp (xs i) / ∑ j ∈ Finset.range n, Real.exp (xs j)

end
end VeriTile.Bench.Repairs.SoftmaxTriton1.RealModel

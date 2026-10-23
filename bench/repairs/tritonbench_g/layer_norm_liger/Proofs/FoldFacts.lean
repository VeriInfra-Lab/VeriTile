import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.BackwardModel
namespace VeriTile.Bench.Repairs.LayerNormLiger.Folds
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section

theorem project {α β : Type} (update : Nat → α → α) (small : Nat → β → β)
    (P : α → Prop) (value : α → β)
    (hstep : ∀ n a, P a → P (update n a))
    (hvalue : ∀ n a, P a → value (update n a) = small n (value a))
    (cur stop : Nat) (a : α) (ha : P a) :
    value (foldRange update cur stop 1 a) = foldRange small cur stop 1 (value a) := by
  by_cases hc : cur < stop
  · conv_lhs => rw [foldRange]
    conv_rhs => rw [foldRange]
    simp only [Nat.one_ne_zero, hc, if_false, if_true, dite_false, dite_true]
    rw [project update small P value hstep hvalue (cur+1) stop (update cur a) (hstep cur a ha), hvalue cur a ha]
  · simp [foldRange, hc]
termination_by stop-cur
decreasing_by omega

theorem preserves {α : Type} (update : Nat → α → α) (P : α → Prop)
    (hstep : ∀ n a, P a → P (update n a)) (cur stop : Nat) (a : α) (ha : P a) :
    P (foldRange update cur stop 1 a) := by
  by_cases hc : cur < stop
  · rw [foldRange]
    simp only [Nat.one_ne_zero, hc, if_false, if_true, dite_false, dite_true]
    exact preserves update P hstep (cur+1) stop (update cur a) (hstep cur a ha)
  · simpa [foldRange, hc] using ha
termination_by stop-cur
decreasing_by omega

theorem add_sum (f : Nat → ℝ) (cur stop : Nat) (a : ℝ) :
    foldRange (fun n z => z+f n) cur stop 1 a = a + ∑ n ∈ Finset.Ico cur stop, f n := by
  by_cases hc : cur < stop
  · rw [foldRange]
    simp only [Nat.one_ne_zero, hc, if_false, if_true, dite_false, dite_true]
    rw [add_sum f (cur+1) stop (a+f cur), Finset.sum_eq_sum_Ico_succ_bot hc]
    ring
  · simp [foldRange, hc, Finset.Ico_eq_empty_of_le (Nat.le_of_not_gt hc)]
termination_by stop-cur
decreasing_by omega

theorem set_outside {α : Type} (row cur stop : Nat) (target a : α)
    (h : row < cur ∨ stop ≤ row) :
    foldRange (fun n z => if n=row then target else z) cur stop 1 a = a := by
  by_cases hc : cur < stop
  · rw [foldRange]
    have hn : cur ≠ row := by omega
    simp only [Nat.one_ne_zero, hc, if_false, if_true, dite_false, dite_true, hn]
    exact set_outside row (cur+1) stop target a (by omega)
  · simp [foldRange, hc]
termination_by stop-cur
decreasing_by omega

theorem set_inside {α : Type} (row cur stop : Nat) (target a : α)
    (lo : cur ≤ row) (hi : row < stop) :
    foldRange (fun n z => if n=row then target else z) cur stop 1 a = target := by
  have hc : cur < stop := by omega
  rw [foldRange]
  simp only [Nat.one_ne_zero, hc, if_false, if_true, dite_false, dite_true]
  by_cases hn : cur=row
  · simp only [hn, if_true]
    exact set_outside row (row+1) stop target target (Or.inl (by omega))
  · simp only [hn, if_false]
    exact set_inside row (cur+1) stop target a (by omega) hi
termination_by stop-cur
decreasing_by omega

#axiomsClean project
#axiomsClean preserves
#axiomsClean add_sum
#axiomsClean set_inside
#axiomsClean set_outside
end
end VeriTile.Bench.Repairs.LayerNormLiger.Folds

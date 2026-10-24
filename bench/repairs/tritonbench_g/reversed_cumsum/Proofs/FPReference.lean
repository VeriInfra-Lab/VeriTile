import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.FPExecution

namespace VeriTile.Bench.Repairs.ReversedCumsum.FPReference
open VeriTile.Triton FP.Structural FPValues FPExecution
noncomputable section
variable {α : Type} [Inhabited α]

theorem loaded_congr (M : Algebra α) (h : Call) (c : Config) (f : Nat) (a b : Nat → Nat → α)
    (he : ∀ t s, t<h.T → s<h.S → a t s=b t s) (q : Nat) : loaded M h c f a q=loaded M h c f b q := by
  funext i
  unfold loaded
  by_cases hv : q*c.BT+i.1.val<h.T ∧ f*BS+i.2.1.val<h.S
  · simp [hv,he _ _ hv.1 hv.2]
  · simp [hv]

theorem carry_congr (M : Algebra α) (h : Call) (c : Config) (f : Nat) (a b : Nat → Nat → α)
    (he : ∀ t s, t<h.T → s<h.S → a t s=b t s) (q : Nat) : carry M h c f a q=carry M h c f b q := by
  conv_lhs => rw [carry]
  conv_rhs => rw [carry]
  by_cases hq : q<Geometry.chunks h c
  · simp only [if_pos hq]
    unfold FPValues.updated
    rw [loaded_congr M h c f a b he q,carry_congr M h c f a b he (q+1)]
  · simp only [if_neg hq]
termination_by Geometry.chunks h c-q

theorem expected_congr (M : Algebra α) (dot : DotBackend α) (h : Call) (c : Config) (positive : 0<c.BT)
    (f : Nat) (a b : Nat → Nat → α) (he : ∀ t s, t<h.T → s<h.S → a t s=b t s) (t : Nat) (j : Fin BS) :
    expected M dot h c positive f a t j=expected M dot h c positive f b t j := by
  unfold expected FPValues.output
  rw [loaded_congr M h c f a b he,carry_congr M h c f a b he]

#print axioms expected_congr
end
end VeriTile.Bench.Repairs.ReversedCumsum.FPReference

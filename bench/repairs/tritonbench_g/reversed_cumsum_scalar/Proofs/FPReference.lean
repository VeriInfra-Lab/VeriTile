import bench.repairs.tritonbench_g.reversed_cumsum_scalar.Proofs.FPExecution

namespace VeriTile.Bench.Repairs.ReversedCumsumScalar.FPReference
open VeriTile.Triton FP.Structural FPValues FPExecution
noncomputable section
variable {α : Type} [Inhabited α]

theorem loaded_congr (M : Algebra α) (h : Call) (c : Config) (a b : Nat → α)
    (he : ∀ t, t<h.T → a t=b t) (q : Nat) : loaded M h c a q=loaded M h c b q := by
  funext i
  unfold loaded
  by_cases ht : q*c.BT+i.1.val<h.T <;> simp [ht,he]

theorem carry_congr (M : Algebra α) (h : Call) (c : Config) (a b : Nat → α)
    (he : ∀ t, t<h.T → a t=b t) (q : Nat) : carry M h c a q=carry M h c b q := by
  conv_lhs => rw [carry]
  conv_rhs => rw [carry]
  by_cases hq : q<Geometry.chunks h c
  · simp only [if_pos hq,updated,total,loaded_congr M h c a b he q,
      carry_congr M h c a b he (q+1)]
  · simp only [if_neg hq]
termination_by Geometry.chunks h c-q

theorem expected_congr (M : Algebra α) (scan : ScanBackend α) (h : Call) (c : Config)
    (positive : 0<c.BT) (a b : Nat → α) (he : ∀ t, t<h.T → a t=b t) (time : Nat) :
    expected M scan h c positive a time=expected M scan h c positive b time := by
  unfold expected output
  rw [loaded_congr M h c a b he,carry_congr M h c a b he]

#print axioms expected_congr
end
end VeriTile.Bench.Repairs.ReversedCumsumScalar.FPReference

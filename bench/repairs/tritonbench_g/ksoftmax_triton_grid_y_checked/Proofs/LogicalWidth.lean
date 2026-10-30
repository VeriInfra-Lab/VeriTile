import bench.repairs.tritonbench_g.ksoftmax_triton_grid_y_checked.Proofs.RealIO

namespace VeriTile.Bench.Repairs.KsoftmaxGridYChecked.LogicalWidth
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.Ksoftmax
open VeriTile.Bench.TritonBenchG.KsoftmaxTriton
open scoped BigOperators
noncomputable section

/-- Removing the padded zero tail leaves a sum over exactly the logical K
coordinates. Coverage supplies the embedding; no input restriction is added. -/
theorem sum_restrict {K D : Nat} (covered : K≤D) (f : Fin D → ℝ)
    (tail : ∀ j, K≤j.val → f j=0) :
    (∑ j : Fin D, f j) = ∑ j : Fin K, f (Fin.castLE covered j) := by
  symm
  apply Fintype.sum_of_injective (Fin.castLE covered) (Fin.castLE_injective covered)
  · intro j outside
    apply tail j
    by_contra h
    have small : j.val<K := Nat.lt_of_not_ge h
    exact outside ⟨⟨j.val,small⟩,Fin.ext rfl⟩
  · intro j
    rfl

theorem readActive_tail (K row : Nat) (causal : Bool) (j : Nat) (tail : K≤j) :
    Forward.readActive K row causal j = Bool.false := by
  cases causal <;> simp [Forward.readActive,Nat.not_lt.mpr tail]

theorem backwardActive_tail (K row : Nat) (causal : Bool) (j : Nat) (tail : K≤j) :
    Backward.readActive K row causal j = Bool.false := by
  cases causal <;> simp [Backward.readActive,Nat.not_lt.mpr tail]

theorem scores_tail {B Q K D : Nat} (causal : Bool) (mode : Forward.MaskMode)
    (v : Interface.Inputs (WithBot ℝ) B Q D) (b : Fin B) (r : Fin Q) (j : Fin D)
    (tail : K≤j.val) : Interface.forwardScores K causal mode v b r j=none := by
  simp [Interface.forwardScores,KernelExtended.scores,readActive_tail K r causal j.val tail]

#axiomsClean sum_restrict
#axiomsClean scores_tail
end
end VeriTile.Bench.Repairs.KsoftmaxGridYChecked.LogicalWidth

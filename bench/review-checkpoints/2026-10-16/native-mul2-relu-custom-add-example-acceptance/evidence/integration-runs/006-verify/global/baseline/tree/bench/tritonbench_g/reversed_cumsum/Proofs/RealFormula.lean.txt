import bench.tritonbench_g.reversed_cumsum.Proofs.RealInit
import bench.tritonbench_g.reversed_cumsum.Proofs.Formula

/-! The loop invariant's independently defined result is the conventional
input suffix plus exactly the source's unspecified time-tail contribution. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsum.RealFormula
open VeriTile.Triton RealProgram RealStep
open scoped BigOperators
noncomputable section

def suffix (p : Layout) (l : Launch) (a : Nat → ℝ) (t : Nat) (j : Fin p.BS) : ℝ :=
  ∑ k ∈ (Finset.range p.T).filter (fun k => t ≤ k), a (address p l k j.val)

def tail (p : Layout) (l : Launch) (padding : Nat → ℝ) (j : Fin p.BS) : ℝ :=
  Formula.tail (fun k => padding (address p l k j.val)) p.T (numChunks p * p.BT)

theorem expected_eq (p : Layout) (l : Launch) (a padding : Nat → ℝ)
    (t : Nat) (j : Fin p.BS) (hBT : 0 < p.BT) (ht : t < p.T)
    (hj : l.feature * p.BS + j.val < p.S) :
    RealLoop.expected p l a padding t j = suffix p l a t j + tail p l padding j := by
  have hcol : (fun k => value p l a padding k j) =
      Formula.paddedInput (fun k => a (address p l k j.val))
        (fun k => padding (address p l k j.val)) p.T := by
    funext k
    simp only [value, active, hj, and_true, Formula.paddedInput]
  unfold RealLoop.expected
  rw [hcol, Formula.padded_suffix _ _ _ _ _ (RealInit.chunks_cover p hBT) (Nat.le_of_lt ht)]
  rw [RealMath.suffix_filter _ _ _ (Nat.le_of_lt ht)]
  rfl

theorem zero_oracle (p : Layout) (l : Launch) (a padding : Nat → ℝ)
    (t : Nat) (j : Fin p.BS) (hBT : 0 < p.BT) (ha : active p l t j.val)
    (hz : ∀ offset, padding offset = 0) :
    RealLoop.expected p l a padding t j = suffix p l a t j := by
  rw [expected_eq p l a padding t j hBT ha.1 ha.2]
  simp [tail, Formula.tail, hz]

#axiomsClean expected_eq
#axiomsClean zero_oracle
end
end VeriTile.Bench.TritonBenchG.ReversedCumsum.RealFormula

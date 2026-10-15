import bench.tritonbench_g.reversed_cumsum_scalar.Proofs.RealInit

/-! The actual source result is the ordinary input suffix plus precisely the
unspecified padded tail sum. The ordinary formula requires zero total tail. -/
namespace VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealFormula
open VeriTile.Triton RealProgram RealValues
open scoped BigOperators
noncomputable section

def suffix (T row : Nat) (input : Nat → ℝ) (t : Nat) : ℝ :=
  ∑ k ∈ (Finset.range T).filter (fun k => t ≤ k), input (address T row k)

def tail (T BT row : Nat) (padding : Nat → ℝ) : ℝ :=
  Formula.tail (fun k => padding (address T row k)) T (numChunks T BT * BT)

theorem expected_eq (T BT row : Nat) (input padding : Nat → ℝ)
    (t : Nat) (hBT : 0 < BT) (ht : t < T) :
    RealLoop.expected T BT row input padding t =
      suffix T row input t + tail T BT row padding := by
  change Formula.suffix (Formula.padded (fun k => input (address T row k))
    (fun k => padding (address T row k)) T) (numChunks T BT * BT) t = _
  rw [Formula.padding_contribution _ _ _ _ _ (RealInit.chunks_cover T BT hBT) (Nat.le_of_lt ht)]
  rw [Formula.suffix_filter _ _ _ (Nat.le_of_lt ht)]
  rfl

theorem ordinary_suffix_iff (T BT row : Nat) (input padding : Nat → ℝ)
    (t : Nat) (hBT : 0 < BT) (ht : t < T) :
    RealLoop.expected T BT row input padding t = suffix T row input t ↔
      tail T BT row padding = 0 := by
  rw [expected_eq T BT row input padding t hBT ht]
  constructor <;> intro h <;> linarith

theorem zero_oracle (T BT row : Nat) (input padding : Nat → ℝ)
    (t : Nat) (hBT : 0 < BT) (ht : t < T) (hz : ∀ offset, padding offset = 0) :
    RealLoop.expected T BT row input padding t = suffix T row input t := by
  rw [expected_eq T BT row input padding t hBT ht]
  simp [tail, Formula.tail, hz]

theorem aligned (T BT row : Nat) (input padding : Nat → ℝ)
    (t : Nat) (hBT : 0 < BT) (ht : t < T) (halign : numChunks T BT * BT = T) :
    RealLoop.expected T BT row input padding t = suffix T row input t := by
  rw [ordinary_suffix_iff T BT row input padding t hBT ht]
  unfold tail
  rw [halign, Formula.tail_zero_of_aligned]

#axiomsClean expected_eq
#axiomsClean ordinary_suffix_iff
#axiomsClean zero_oracle
#axiomsClean aligned
end
end VeriTile.Bench.TritonBenchG.ReversedCumsumScalar.RealFormula

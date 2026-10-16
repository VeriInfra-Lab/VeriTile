import bench.tritonbench_g.int8_matmul_kernel.Proofs.WordReference
namespace VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel
open VeriTile.Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

theorem extracted_field (w sh : Nat) : (w &&& 3 <<< sh) >>> sh = w >>> sh &&& 3 := by
  apply Nat.eq_of_testBit_eq
  intro n
  simp [Nat.testBit_shiftRight, Nat.testBit_and, Nat.testBit_shiftLeft]

theorem wrap8_bounds (x : Int) : -128 ≤ wrap 8 x ∧ wrap 8 x < 128 := by
  have hl := @BitVec.le_two_mul_toInt 8 (BitVec.ofInt 8 x)
  have hu := @BitVec.toInt_lt 8 (BitVec.ofInt 8 x)
  change -128 ≤ (BitVec.ofInt 8 x).toInt ∧ (BitVec.ofInt 8 x).toInt < 128
  norm_num only [Nat.reducePow, Nat.reduceSub, Int.reducePow] at hl hu
  omega

theorem a_signed8 {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (f k : Nat) (i : AIndex p) : FP.IntegerDot.signed8 (aValue p s f k i) :=
  wrap8_bounds _

theorem b_signed8 {α : Type} [Inhabited α] (p : Parameters) (s : State α)
    (f k : Nat) (i : BIndex p) : -1 ≤ bValue p s f k i ∧ bValue p s f k i ≤ 2 := by
  dsimp only [bValue]
  rw [extracted_field]
  let n : Nat := ((if i.1.val < p.inner then
    (s.mem p.weight (bOffset p s k i)).read .nat else 0) >>> (2*f)) &&& 3
  have hn : n ≤ 3 := Nat.and_le_right
  change -1 ≤ wrap 8 (n : Int) - 1 ∧ wrap 8 (n : Int) - 1 ≤ 2
  have hc : (n : Int) ≤ 3 := Int.ofNat_le.mpr hn
  have he : wrap 8 (n : Int) = (n : Int) := by
    apply BitVec.toInt_ofInt_eq_self (by decide) <;> omega
  rw [he]
  omega

/-- The dot callback always receives actual signed-byte values, including
negative A values produced by truncating the original int32 storage. -/
theorem dot_inputs_valid {α : Type} [Inhabited α] (p : Parameters) (s : State α) (f k : Nat) :
    FP.IntegerDot.valid (α := α) (aValue p s f k) ∧ FP.IntegerDot.valid (α := α) (bValue p s f k) := by
  constructor
  · exact a_signed8 p s f k
  · intro i
    have h := b_signed8 p s f k i
    exact ⟨by omega, by omega⟩

theorem narrowing_changes_product :
    wordDot (m := 1) (k := 1) (n := 1) (fun _ => wrap 8 255) (fun _ => wrap 8 3 - 1)
      (0,0,PUnit.unit) = -2 ∧ (255 * (3-1) : Int) = 510 := by decide

theorem accumulation_wraps : wrap 32 (2147483647+1) = -2147483648 := by decide

#axiomsClean extracted_field
#axiomsClean wrap8_bounds
#axiomsClean a_signed8
#axiomsClean b_signed8
#axiomsClean dot_inputs_valid
#axiomsClean narrowing_changes_product
#axiomsClean accumulation_wraps
end
end VeriTile.Bench.TritonBenchG.Int8MatmulKernel.WordModel

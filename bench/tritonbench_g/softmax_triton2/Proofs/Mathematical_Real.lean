import VeriTile.Triton.DSL
import VeriTile.Triton
import VeriTile.Triton.Semantics.MaskedReduction

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton2
open VeriTile.Triton

/-- Masked input row tile used by `softmax_kernel`: lane `j < n_cols` holds
`xs j`, masked lanes are `⊥`, matching `other=-float("inf")`. -/
noncomputable def softmaxInputTile (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) : Tile .real [BLOCK_SIZE] :=
  { data := fun idx => if idx.1.val < n_cols then some (xs idx.1) else none }

/-- Exact stable-softmax value computed by the kernel at lane `idx`, as a pure
function of the active row prefix `xs j`, `j < n_cols` (masked lanes enter the
reductions as `⊥`, neutral for both `max` and the `exp`-sum). -/
noncomputable def softmaxSpec (n_cols BLOCK_SIZE : Nat)
    (xs : Fin BLOCK_SIZE → ℝ) (idx : Fin BLOCK_SIZE) : ℝ :=
  let row := softmaxInputTile n_cols BLOCK_SIZE xs
  match Tile.reduceMax (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false row with
  | some rowMax =>
      let shifted := Tile.bop (NumericDType.sub .real) Broadcast.scalarR row rowMax
      let numerator := Tile.uop WithBot.realExp shifted
      let denominator := Tile.reduceSum (shape := [BLOCK_SIZE]) ⟨0, by simp⟩ Bool.false numerator
      WithBot.unbotD 0
        ((Tile.bop (NumericDType.div .real) Broadcast.scalarR numerator denominator).data
          (idx, PUnit.unit))
  | none => 0

/-- `softmaxSpec` only reads the active lanes of its input: two rows agreeing
below `n_cols` yield the same softmax value (masked lanes are `⊥` in the tile
either way). -/
theorem softmaxSpec_congr (n_cols BLOCK_SIZE : Nat) (xs ys : Fin BLOCK_SIZE → ℝ)
    (h : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → xs j = ys j) (i : Fin BLOCK_SIZE) :
    softmaxSpec n_cols BLOCK_SIZE xs i = softmaxSpec n_cols BLOCK_SIZE ys i := by
  have htile : softmaxInputTile n_cols BLOCK_SIZE xs
      = softmaxInputTile n_cols BLOCK_SIZE ys := by
    unfold softmaxInputTile
    congr 1
    funext idx
    by_cases hj : idx.1.val < n_cols
    · simp only [if_pos hj, h idx.1 hj]
    · simp only [if_neg hj]
  unfold softmaxSpec
  rw [htile]

/-- The masked tile computation agrees with softmax over exactly the active lanes. -/
theorem softmaxSpec_eq_formula (n B : Nat) (hB : 0 < B)
    (xs : Fin B → ℝ) (i : Fin B) (hi : i.val < n) :
    softmaxSpec n B xs i =
      Real.exp (xs i) / ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0 := by
  classical
  obtain ⟨k, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hB.ne'
  have hn : ((Finset.univ : Finset (Fin (k + 1))).filter (fun j => j.val < n)).Nonempty :=
    ⟨i, by simp [hi]⟩
  have hm := TiledLogSumExp.withBot_sup'_partial (α := ℝ) Finset.univ
    Finset.univ_nonempty (fun j : Fin (k + 1) => j.val < n) xs hn
  simp [softmaxSpec, softmaxInputTile, Tile.reduceMax, Tile.reduceMaxDrop,
    Tile.reduceSum, Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
    TileShape.insertAxisIndex, Tile.bop, Tile.uop, NumericDType.sub,
    NumericDType.div, hi]
  have hm' : @Finset.sup' (WithBot ℝ) (Fin (k + 1)) _ Finset.univ Finset.univ_nonempty
      (fun j : Fin (k + 1) => if j.val < n then (some (xs j) : WithBot ℝ) else none) =
      (some ((Finset.univ.filter (fun j : Fin (k + 1) => j.val < n)).sup' hn xs) : WithBot ℝ) := hm
  erw [hm']
  let m := (Finset.univ.filter (fun j : Fin (k + 1) => j.val < n)).sup' hn xs
  have hsum := TiledLogSumExp.sum_exp_masked_eq (n := k) (D := n) (i_d := 0) xs m
  simp only [Nat.zero_mul, Nat.zero_add] at hsum
  erw [hsum]
  change Real.exp (xs i - m) /
      (∑ j ∈ TiledLogSumExp.validLanes k n 0, Real.exp (xs j - m)) = _
  simp only [TiledLogSumExp.validLanes, Nat.zero_mul, Nat.zero_add,
    Finset.sum_filter, Real.exp_sub]
  have hs : ∀ j : Fin (k + 1),
      (if j.val < n then Real.exp (xs j) / Real.exp m else 0) =
      (if j.val < n then Real.exp (xs j) else 0) / Real.exp m := by
    intro j
    split_ifs <;> simp
  simp_rw [hs]
  rw [← Finset.sum_div]
  exact div_div_div_cancel_right₀ (Real.exp_ne_zero m) _ _

private theorem mul_recip (a b : Option ℝ) :
    Option.map₂ (· * ·) a (b.map (fun x => x⁻¹)) = Option.map₂ (· / ·) a b := by
  cases a <;> cases b <;> simp [div_eq_mul_inv]

end VeriTile.Bench.TritonBenchG.SoftmaxTriton2

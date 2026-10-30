import bench.repairs.tritonbench_g.attention_score_tile_checked.Proofs.Reused.RealLoop

/-! The independent logical target sums precisely N queries. It contains no
block size, launch flags, pointer padding, execution state or assumed producer
for the supplied normalizer. The later lemmas connect the repaired loop to it. -/
namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.LogicalFormula
open VeriTile.Triton RealFormula
noncomputable section

def weight {D : Nat} (xs : Inputs D) (scale : ℝ)
    (window complement : Bool) (offset size : Int) (row key : Nat) : ℝ :=
  if windowAllows window complement offset size row key then
    pow2 ((∑ d : Fin D, xs.query row d * xs.key key d) *
      (scale * 1.4426950408889634) - xs.normalizer row)
  else 0

def column {D : Nat} (xs : Inputs D) (scale : ℝ) (N : Nat)
    (window complement : Bool) (offset size : Int) (key : Nat) : ℝ :=
  ∑ row : Fin N, weight xs scale window complement offset size row.val key

def hostScale (D : Nat) : ℝ := 1 / Real.sqrt D

theorem partition (f : Nat → ℝ) (n B : Nat) :
    (∑ b : Fin n, ∑ r : Fin B, f (b.val * B + r.val)) =
      ∑ row ∈ Finset.range (n * B), f row := by
  induction n with
  | zero => simp
  | succ n ih =>
      rw [Fin.sum_univ_castSucc]
      simp only [Fin.val_castSucc, Fin.val_last]
      rw [ih, Nat.succ_mul, Finset.sum_range_add]
      congr 1
      exact Fin.sum_univ_eq_sum_range (fun r => f (n * B + r)) B

theorem exact_cover (N B : Nat) (hB : 0 < B) (hEven : N % B = 0) :
    RealLoop.tripCount N B * B = N := by
  have hd : B * (N / B) = N := by
    simpa [hEven] using Nat.div_add_mod N B
  have hcount : RealLoop.tripCount N B ≤ N / B :=
    (ceilDiv_le_iff_le_mul hB).2 (le_of_eq hd.symm)
  have hcover := (RealLoop.tripCount_bounds N B hB).1
  have hmul := Nat.mul_le_mul_right B hcount
  rw [Nat.mul_comm (N / B) B, hd] at hmul
  exact Nat.le_antisymm hmul hcover

theorem active_weight {D : Nat} (xs : Inputs D) (scale : ℝ)
    (N NK row key : Nat) (window complement evenM evenN : Bool)
    (offset size : Int) (hr : row < N) (hk : key < NK) :
    RealFormula.weight xs scale N NK window complement evenM evenN offset size row key =
      weight xs scale window complement offset size row key := by
  simp [RealFormula.weight, RealFormula.score, weight, hr, hk]

/-- The only even-query fact needed here is the actual host modulus test.
There is no M-padding condition and no premise about the final column value. -/
theorem repaired_column {D : Nat} (xs : Inputs D) (scale : ℝ)
    (N NK B key : Nat) (window complement evenM evenN : Bool) (offset size : Int)
    (hB : 0 < B) (hk : key < NK) (he : evenM = Bool.true → N % B = 0) :
    RealFormula.column xs scale N NK B (RealLoop.tripCount N B)
      window complement evenM evenN offset size key =
      column xs scale N window complement offset size key := by
  unfold RealFormula.column
  rw [partition (fun row => RealFormula.weight xs scale N NK window complement evenM evenN offset size row key)]
  let f := fun row => RealFormula.weight xs scale N NK window complement evenM evenN offset size row key
  have htail : ∀ row, row < RealLoop.tripCount N B * B → N ≤ row → f row = 0 := by
    intro row hb hn
    cases hem : evenM
    · simp [f, RealFormula.weight, hem, Nat.not_lt.mpr hn]
    · have hc := exact_cover N B hB (he hem)
      omega
  have hcut : (∑ row ∈ Finset.range (RealLoop.tripCount N B * B), f row) =
      ∑ row ∈ Finset.range N, f row := by
    symm
    apply Finset.sum_subset (Finset.range_mono (RealLoop.tripCount_bounds N B hB).1)
    intro row hb hn
    exact htail row (Finset.mem_range.mp hb) (by simpa using hn)
  change (∑ row ∈ Finset.range (RealLoop.tripCount N B * B), f row) = _
  rw [hcut]
  unfold column
  rw [Fin.sum_univ_eq_sum_range (fun row => weight xs scale window complement offset size row key)]
  apply Finset.sum_congr rfl
  intro row hr
  exact active_weight xs scale N NK row key window complement evenM evenN offset size
    (Finset.mem_range.mp hr) hk

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.LogicalFormula

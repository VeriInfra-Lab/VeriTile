import VeriTile.Triton.KernelLemmas.EvalHelpers

/-!
Arithmetic of the original rank-one wrapper's capped launch. This module proves
coverage and uniqueness of logical indices, independently of scalar precision.
It does not establish signed block-pointer lowering or native integer bounds.
The wrapper's empty-input failure is outside the positive-size statements.
-/

namespace VeriTile.Bench.TritonBenchG.ReluStridedBuffer.CappedGrid

open VeriTile.Triton

def numTiles (n b : Nat) : Nat := cdiv n b
def numCTAs (n b : Nat) : Nat := min 65536 (numTiles n b)
def tilesPerCTA (n b : Nat) : Nat := cdiv (numTiles n b) (numCTAs n b)
def logicalIndex (b c pid step lane : Nat) : Nat := (pid + step * c) * b + lane

theorem cdiv_covers {n b : Nat} (hb : 0 < b) : n ≤ cdiv n b * b := by
  have h := (Nat.div_lt_iff_lt_mul hb).mp (Nat.lt_succ_self ((n + b - 1) / b))
  simp only [Nat.succ_mul] at h
  unfold cdiv
  omega

theorem cdiv_pos {n b : Nat} (hn : 0 < n) (hb : 0 < b) : 0 < cdiv n b := by
  have h := cdiv_covers (n := n) hb
  by_contra hz
  have : cdiv n b = 0 := by omega
  rw [this] at h
  simp at h
  omega

theorem cdiv_eq_one_iff {n b : Nat} (hn : 0 < n) (hb : 0 < b) :
    cdiv n b = 1 ↔ n ≤ b := by
  have hp := cdiv_pos hn hb
  have hl := Nat.div_le_iff_le_mul (x := n + b - 1) (y := 1) hb
  unfold cdiv at *
  simp only [Nat.one_mul] at hl
  omega

theorem numCTAs_pos {n b : Nat} (hn : 0 < n) (hb : 0 < b) :
    0 < numCTAs n b := by
  unfold numCTAs numTiles
  exact lt_min (by omega) (cdiv_pos hn hb)

theorem one_tile_iff {n b : Nat} (hn : 0 < n) (hb : 0 < b) :
    tilesPerCTA n b = 1 ↔ numTiles n b ≤ 65536 := by
  have hp : 0 < numTiles n b := cdiv_pos hn hb
  rw [tilesPerCTA, cdiv_eq_one_iff hp (numCTAs_pos hn hb)]
  simp [numCTAs]

/-- For each source-admitted logical element, division selects a legal CTA,
loop iteration and lane whose actual source index is exactly that element. -/
theorem logical_index_covered {n b i : Nat} (hb : 0 < b) (hi : i < n) :
    let c := numCTAs n b
    let q := i / b
    q % c < c ∧ q / c < tilesPerCTA n b ∧ i % b < b ∧
      logicalIndex b c (q % c) (q / c) (i % b) = i := by
  dsimp
  have hn : 0 < n := by omega
  have hc := numCTAs_pos hn hb
  have hq : i / b < numTiles n b := by
    apply (Nat.div_lt_iff_lt_mul hb).mpr
    exact Nat.lt_of_lt_of_le hi (cdiv_covers hb)
  have ht : i / b / numCTAs n b < tilesPerCTA n b := by
    apply (Nat.div_lt_iff_lt_mul hc).mpr
    exact Nat.lt_of_lt_of_le hq (cdiv_covers hc)
  refine ⟨Nat.mod_lt _ hc, ht, Nat.mod_lt _ hb, ?_⟩
  unfold logicalIndex
  rw [Nat.mul_comm (i / b / numCTAs n b), Nat.mod_add_div,
    Nat.mul_comm (i / b), Nat.div_add_mod]

/-- Two valid CTA/lane coordinates cannot write the same logical position.
This holds even for the masked-off iterations at the end of the capped grid. -/
theorem logical_index_injective {b c p j l p' j' l' : Nat}
    (hb : 0 < b) (hc : 0 < c)
    (hp : p < c) (hp' : p' < c) (hl : l < b) (hl' : l' < b)
    (h : logicalIndex b c p j l = logicalIndex b c p' j' l') :
    p = p' ∧ j = j' ∧ l = l' := by
  have hmod := congrArg (fun v => v % b) h
  simp only [logicalIndex, Nat.mul_add_mod_self_right, Nat.mod_eq_of_lt hl,
    Nat.mod_eq_of_lt hl'] at hmod
  have htile : p + j * c = p' + j' * c := by
    unfold logicalIndex at h
    have hm : (p + j * c) * b = (p' + j' * c) * b := by omega
    exact Nat.eq_of_mul_eq_mul_right hb hm
  have hpids := congrArg (fun v => v % c) htile
  simp only [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hp,
    Nat.mod_eq_of_lt hp'] at hpids
  have hj : j = j' := by
    have hm : j * c = j' * c := by omega
    exact Nat.eq_of_mul_eq_mul_right hc hm
  exact ⟨hpids, hj, hmod⟩

/-- The one-tile branch uses the same logical indices as iteration zero of the
grid-stride branch. Combined with coverage, this connects the wrapper switch. -/
theorem one_tile_coverage {n b i : Nat} (hb : 0 < b) (hi : i < n)
    (hone : tilesPerCTA n b = 1) :
    i / b < numCTAs n b ∧ logicalIndex b (numCTAs n b) (i / b) 0 (i % b) = i := by
  have h := logical_index_covered hb hi
  dsimp at h
  rw [hone] at h
  have hzero : i / b / numCTAs n b = 0 := Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ h.2.1)
  have hc := numCTAs_pos (by omega : 0 < n) hb
  have hp : i / b < numCTAs n b :=
    (Nat.div_eq_zero_iff.mp hzero).resolve_left (Nat.ne_of_gt hc)
  refine ⟨hp, ?_⟩
  simpa [logicalIndex, Nat.mod_eq_of_lt hp, hzero] using h.2.2.2

end VeriTile.Bench.TritonBenchG.ReluStridedBuffer.CappedGrid

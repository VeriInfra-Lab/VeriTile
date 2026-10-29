import Mathlib.Tactic

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.FlatSchedule

/-- Tile register before flat iteration i, retaining the signed pre-loop value. -/
def previousTile (sms pid k i : Nat) : Int :=
  if i = 0 then (pid : Int) - (sms : Int) else (pid + ((i-1)/k)*sms : Nat)

/-- Quotient recurrence for the exact flattened persistent trip count. -/
theorem quotient_next (k i : Nat) (hk : 0 < k) :
    (i+1)/k = i/k + (i%k+1)/k := by
  have hi : i+1 = (i%k+1)+(i/k)*k := by
    have h := Nat.mod_add_div i k
    rw [Nat.mul_comm k] at h
    omega
  rw [hi,Nat.add_mul_div_right _ _ hk,Nat.add_comm]

theorem last_iff_restart (k i : Nat) (hk : 0 < k) :
    i%k+1 = k ↔ (i+1)%k = 0 := by
  rw [← Nat.mod_add_mod i k 1]
  have hm := Nat.mod_lt i hk
  constructor
  · intro h
    rw [h,Nat.mod_self]
  · intro h
    by_contra hn
    rw [Nat.mod_eq_of_lt (show i%k+1 < k by omega)] at h
    omega

theorem quotient_last (k i : Nat) (hk : 0 < k) (h : i%k+1 = k) :
    (i+1)/k = i/k+1 := by
  rw [quotient_next k i hk,h,Nat.div_self hk]

theorem quotient_continue (k i : Nat) (hk : 0 < k) (h : i%k+1 ≠ k) :
    (i+1)/k = i/k := by
  have hm := Nat.mod_lt i hk
  rw [quotient_next k i hk,Nat.div_eq_of_lt (show i%k+1 < k by omega),Nat.add_zero]

/-- On every K restart the source advances from the previous signed tile to
exactly the output ordinal i/k, including i=0. -/
theorem tile_restart (sms pid k i : Nat) (hk : 0 < k) (hi : i%k = 0) :
    previousTile sms pid k i = (pid+(i/k)*sms : Nat) - (sms : Int) := by
  cases i with
  | zero => simp [previousTile]
  | succ i =>
    have hlast := (last_iff_restart k i hk).mpr hi
    have hq := quotient_last k i hk hlast
    simp only [previousTile,Nat.succ_ne_zero,if_false,Nat.succ_sub_one,Nat.succ_eq_add_one,hq]
    push_cast
    ring

/-- Between K restarts the source keeps the tile for the same output ordinal. -/
theorem tile_continue (sms pid k i : Nat) (hk : 0 < k) (hi : i%k ≠ 0) :
    previousTile sms pid k i = (pid+(i/k)*sms : Nat) := by
  cases i with
  | zero => simp at hi
  | succ i =>
    have hlast : i%k+1 ≠ k := by
      intro h
      exact hi ((last_iff_restart k i hk).mp h)
    have hq := quotient_continue k i hk hlast
    simp [previousTile,Nat.succ_eq_add_one,hq]

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.FlatSchedule

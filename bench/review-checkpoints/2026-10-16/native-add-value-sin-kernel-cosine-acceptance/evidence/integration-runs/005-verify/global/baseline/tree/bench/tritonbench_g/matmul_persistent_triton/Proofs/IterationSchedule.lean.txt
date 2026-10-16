import bench.tritonbench_g.matmul_persistent_triton.Proofs.Schedule

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule

/-- The exact signed where-update gives the next K-block ordinal. -/
theorem ki_next (k i : Nat) (hk : 0 < k) :
    (if (i % k : Nat) = (k : Int) - 1 then (0 : Int) else (i % k : Nat) + 1) =
      ((i + 1) % k : Nat) := by
  have hm := Nat.mod_lt i hk
  rw [← Nat.mod_add_mod i k 1]
  by_cases hlast : i % k + 1 = k
  · have hi : ((i % k : Nat) : Int) = (k : Int) - 1 := by omega
    simp only [hi, if_true, hlast, Nat.mod_self, Nat.cast_zero]
  · have hi : ¬((i % k : Nat) : Int) = (k : Int) - 1 := by omega
    rw [if_neg hi, Nat.mod_eq_of_lt (show i % k + 1 < k by omega)]
    simp

/-- Each source flat-loop iteration belongs to one assigned output tile,
with a K-block index below the real ceiling count. -/
theorem iteration_bounds (k tiles i : Nat) (hk : 0 < k) (hi : i < k * tiles) :
    i / k < tiles ∧ i % k < k := by
  exact ⟨(Nat.div_lt_iff_lt_mul hk).mpr (by simpa [Nat.mul_comm] using hi),
    Nat.mod_lt i hk⟩

/-- Last-K detection is equivalent to the next iteration starting at K=0.
This controls exactly the store/reset and next persistent tile advance. -/
theorem last_iff_restart (k i : Nat) (hk : 0 < k) :
    i % k + 1 = k ↔ (i + 1) % k = 0 := by
  rw [← Nat.mod_add_mod i k 1]
  have hm := Nat.mod_lt i hk
  constructor
  · intro h
    rw [h, Nat.mod_self]
  · intro h
    by_contra hn
    have hl : i % k + 1 < k := by omega
    rw [Nat.mod_eq_of_lt hl] at h
    omega

/-- Signed tile advancement remains the same natural residue-class schedule
at every persistent output boundary, with no lossy signed-to-natural step. -/
theorem next_tile (sms pid q : Nat) :
    (tileId sms pid q : Int) + sms = (tileId sms pid (q + 1) : Int) := by
  simp [tileId, Nat.add_mul]
  omega

end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.Schedule

import bench.tritonbench_g.seeded_dropout.Proofs.Mathematical_SourceDomain

/-! Exact mathematical indexing for the original one-dimensional launch.
These lemmas prove coverage and separation of output indices; they do not
assert that concurrently aliased input/output accesses are race-free. -/
namespace VeriTile.Bench.TritonBenchG.SeededDropout.Index
open SourceDomain

def offset (pid B : Nat) (j : Fin B) : Nat := pid * B + j.val

theorem offset_injective (pid B : Nat) : Function.Injective (offset pid B) := by
  intro i j h
  apply Fin.ext
  exact Nat.add_left_cancel h

theorem offset_div {B : Nat} (hB : 0 < B) (pid : Nat) (j : Fin B) :
    offset pid B j / B = pid := by
  simpa [offset, Nat.mul_comm, Nat.div_eq_of_lt j.isLt] using
    Nat.mul_add_div hB pid j.val

theorem offset_mod (pid B : Nat) (j : Fin B) :
    offset pid B j % B = j.val := by
  simp [offset, Nat.add_mod, Nat.mod_eq_of_lt j.isLt]

theorem offset_pair_unique {B pid otherPid : Nat} (hB : 0 < B)
    (i j : Fin B) (h : offset pid B i = offset otherPid B j) :
    pid = otherPid ∧ i = j := by
  have hp : pid = otherPid := by
    simpa only [offset_div hB] using congrArg (fun n => n / B) h
  subst otherPid
  exact ⟨rfl, offset_injective pid B h⟩

theorem distinct_programs {B pid otherPid : Nat} (hB : 0 < B)
    (hpid : pid ≠ otherPid) (i j : Fin B) :
    offset pid B i ≠ offset otherPid B j := by
  intro h
  exact hpid (offset_pair_unique hB i j h).1

theorem quotient_lt_grid {N B k : Nat} (hB : 0 < B) (hk : k < N) :
    k / B < grid N B := by
  have hcover : N ≤ B * grid N B :=
    (ceilDiv_le_iff_le_mul hB).mp (show N ⌈/⌉ B ≤ grid N B from le_rfl)
  have hquot : B * (k / B) ≤ k := by
    simpa [Nat.mul_comm] using Nat.div_mul_le_self k B
  by_contra h
  have hmul := Nat.mul_le_mul_left B (Nat.le_of_not_gt h)
  omega

/-- Each original input index has exactly one launched program/lane pair. -/
theorem active_coverage_unique {N B k : Nat} (hB : 0 < B) (hk : k < N) :
    ∃! ij : Fin (grid N B) × Fin B, offset ij.1.val B ij.2 = k := by
  let ij : Fin (grid N B) × Fin B :=
    (⟨k / B, quotient_lt_grid hB hk⟩, ⟨k % B, Nat.mod_lt k hB⟩)
  have heq : offset ij.1.val B ij.2 = k := by
    simpa [offset, ij, Nat.mul_comm] using Nat.div_add_mod k B
  refine ⟨ij, heq, ?_⟩
  intro other ho
  have hpair := offset_pair_unique hB other.2 ij.2 (ho.trans heq.symm)
  exact Prod.ext (Fin.ext hpair.1) hpair.2

theorem active_offset_lt_i32 {N B pid : Nat} (hB : LegalBlock B)
    (hN : N < 2^31) (hpid : pid < grid N B) (j : Fin B) :
    offset pid B j < 2^31 :=
  launched_lane_lt_i32 hB hN hpid j.isLt

end VeriTile.Bench.TritonBenchG.SeededDropout.Index

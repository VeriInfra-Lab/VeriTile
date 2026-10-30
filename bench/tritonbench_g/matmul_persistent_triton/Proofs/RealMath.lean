import bench.tritonbench_g.matmul_persistent_triton.Proofs.Schedule
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealMath
open scoped BigOperators
noncomputable section

/-- Independent input arrays, indexed mathematically rather than read from
an execution result or a prepared accumulator. Only in-bounds indices are
used in the matrix-product formula. -/
structure Inputs where
  a : Nat → Nat → ℝ
  b : Nat → Nat → ℝ

def formula (K : Nat) (xs : Inputs) (row column : Nat) : ℝ :=
  ∑ k ∈ Finset.range K, xs.a row k * xs.b k column

def blocks (length width : Nat) : Nat := (length + width - 1) / width

def partialSum (K BK count : Nat) (xs : Inputs) (row column : Nat) : ℝ :=
  ∑ k ∈ Finset.range (count * BK),
    if k < K then xs.a row k * xs.b k column else 0

def tileSum (K BK iteration : Nat) (xs : Inputs) (row column : Nat) : ℝ :=
  ∑ lane : Fin BK, if lane.val < K - iteration * BK then
    xs.a row (iteration * BK + lane.val) * xs.b (iteration * BK + lane.val) column
  else 0

theorem signed_tail_mask (K BK iteration lane : Nat) :
    (lane : Int) < (K : Int) - (iteration : Int) * BK ↔
      iteration * BK + lane < K := by
  omega

theorem blocks_cover (length width : Nat) (hw : 0 < width) :
    length ≤ blocks length width * width := by
  have h := Nat.lt_mul_div_succ (length + width - 1) hw
  rw [Nat.mul_add, Nat.mul_one, Nat.mul_comm width] at h
  unfold blocks
  omega

theorem blocks_positive (length width : Nat) (hl : 0 < length) (hw : 0 < width) :
    0 < blocks length width := by
  have h := blocks_cover length width hw
  by_contra hz
  have hz' : blocks length width = 0 := by omega
  rw [hz'] at h
  omega

theorem partial_zero (K BK : Nat) (xs : Inputs) (row column : Nat) :
    partialSum K BK 0 xs row column = 0 := by simp [partialSum]

/-- Fused dot's exact-real interpretation extends the previous accumulator
by the current masked K block, including a partialSum final block. -/
theorem partial_succ (K BK count : Nat) (xs : Inputs) (row column : Nat) :
    partialSum K BK (count + 1) xs row column =
      tileSum K BK count xs row column + partialSum K BK count xs row column := by
  simp only [partialSum, tileSum, Nat.add_mul, Nat.one_mul, Finset.sum_range_add]
  rw [add_comm]
  congr 1
  rw [Finset.sum_range]
  apply Finset.sum_congr rfl
  intro lane _
  have heq : lane.val < K - count * BK ↔ count * BK + lane.val < K := by omega
  simp only [heq]

theorem partial_complete (K BK : Nat) (hBK : 0 < BK)
    (xs : Inputs) (row column : Nat) :
    partialSum K BK (blocks K BK) xs row column = formula K xs row column := by
  have hc := blocks_cover K BK hBK
  unfold partialSum formula
  have hs : Finset.range K ⊆ Finset.range (blocks K BK * BK) := Finset.range_mono hc
  have heq := Finset.sum_subset
    (f := fun k => if k < K then xs.a row k * xs.b k column else 0) hs
    (by intro k _ hk; simp only [Finset.mem_range] at hk; simp [hk])
  rw [← heq]
  apply Finset.sum_congr rfl
  intro k hk
  simp only [Finset.mem_range] at hk
  simp [hk]

theorem zero_K (xs : Inputs) (row column : Nat) : formula 0 xs row column = 0 := by
  simp [formula]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealMath

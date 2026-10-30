import bench.tritonbench_g.cross_entropy1.Kernels
import VeriTile.Triton.Math.LogSumExp
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.WithTop

namespace VeriTile.Bench.Repairs.CrossEntropy1.Math
open scoped BigOperators
noncomputable section

def expSum (classes : Nat) (x : Nat → ℝ) : ℝ :=
  ∑ j ∈ Finset.range classes, Real.exp (x j)
def lse (classes : Nat) (x : Nat → ℝ) : ℝ := Real.log (expSum classes x)
def loss (classes : Nat) (x : Nat → ℝ) (target ignored : Int) (smoothing scale : ℝ) : ℝ :=
  if target=ignored then 0 else
    lse classes x-(1-smoothing)*x target.toNat-
      smoothing*(∑ j ∈ Finset.range classes, x j)/classes+scale*lse classes x*lse classes x
def gradient (classes : Nat) (x : Nat → ℝ) (target ignored : Int)
    (smoothing scale upstream : ℝ) (col : Nat) : ℝ :=
  if target=ignored then 0 else
    upstream*((1+2*scale*lse classes x)*Real.exp (x col-lse classes x)-
      (if (col : Int)=target then 1-smoothing else 0)-smoothing/classes)

def blocks (classes width : Nat) : Nat := (classes+width-1)/width
def blockSum (classes width block : Nat) (f : Nat → ℝ) : ℝ :=
  ∑ i ∈ Finset.range width, if block*width+i<classes then f (block*width+i) else 0
def blockLSE (classes width block : Nat) (x : Nat → ℝ) : ℝ :=
  Real.log (blockSum classes width block (fun i => Real.exp (x i)))

theorem expSum_positive (classes : Nat) (x : Nat → ℝ) (positive : 0<classes) :
    0<expSum classes x := by
  apply Finset.sum_pos
  · intro i _
    exact Real.exp_pos _
  · exact ⟨0,by simp [positive]⟩

theorem blocks_cover (classes width : Nat) (positive : 0<width) :
    classes≤blocks classes width*width := by
  have remainder := Nat.mod_lt (classes+width-1) positive
  have decomposition := Nat.div_add_mod (classes+width-1) width
  dsimp [blocks]
  rw [Nat.mul_comm width] at decomposition
  omega

theorem block_starts (classes width block : Nat) (widthPositive : 0<width)
    (launched : block<blocks classes width) : block*width<classes := by
  have bound := Nat.mul_le_mul_right width (Nat.succ_le_of_lt launched)
  have floor := Nat.div_mul_le_self (classes+width-1) width
  dsimp [blocks] at bound
  rw [Nat.succ_mul] at bound
  omega

theorem sum_blocks (count width : Nat) (f : Nat → ℝ) :
    (∑ block ∈ Finset.range count, ∑ lane ∈ Finset.range width, f (block*width+lane))=
      ∑ i ∈ Finset.range (count*width), f i := by
  induction count with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ,ih,Nat.succ_mul,Finset.sum_range_add]

theorem sum_padding (classes capacity : Nat) (bound : classes≤capacity) (f : Nat → ℝ) :
    (∑ i ∈ Finset.range capacity, if i<classes then f i else 0)=
      ∑ i ∈ Finset.range classes, f i := by
  rw [← Finset.sum_filter]
  congr 1
  ext i
  simp only [Finset.mem_filter,Finset.mem_range]
  omega

theorem partition (classes width : Nat) (positive : 0<width) (f : Nat → ℝ) :
    (∑ block ∈ Finset.range (blocks classes width), blockSum classes width block f)=
      ∑ i ∈ Finset.range classes, f i := by
  unfold blockSum
  rw [sum_blocks (blocks classes width) width (fun i => if i<classes then f i else 0)]
  exact sum_padding classes _ (blocks_cover classes width positive) f

theorem block_exp_positive (classes width block : Nat) (positive : 0<width)
    (launched : block<blocks classes width) (x : Nat → ℝ) :
    0<blockSum classes width block (fun i => Real.exp (x i)) := by
  have start := block_starts classes width block positive launched
  apply Finset.sum_pos'
  · intro i _
    split
    · exact le_of_lt (Real.exp_pos _)
    · exact le_refl _
  · refine ⟨0,by simp [positive],?_⟩
    simp only [Nat.add_zero,if_pos start]
    exact Real.exp_pos _

/-- Local padded column blocks compose to the independent full local LSE. -/
theorem local_lse_composition (classes width : Nat) (positive : 0<width) (x : Nat → ℝ) :
    Real.log (∑ block ∈ Finset.range (blocks classes width), Real.exp (blockLSE classes width block x))=
      lse classes x := by
  have each : ∀ block∈Finset.range (blocks classes width),
      Real.exp (blockLSE classes width block x)=blockSum classes width block (fun i => Real.exp (x i)) := by
    intro block hb
    apply Real.exp_log
    exact block_exp_positive classes width block positive (Finset.mem_range.mp hb) x
  simp_rw [Finset.sum_congr rfl each]
  rw [partition classes width positive]
  rfl

/-- Rank concatenation is independent of the local block partition. -/
theorem rank_exp_partition (ranks classes : Nat) (x : Nat → ℝ) :
    (∑ rank ∈ Finset.range ranks, expSum classes (fun j => x (rank*classes+j)))=
      expSum (ranks*classes) x := by
  exact sum_blocks ranks classes (fun i => Real.exp (x i))

theorem rank_lse_composition (ranks classes : Nat) (positive : 0<classes) (x : Nat → ℝ) :
    Real.log (∑ rank ∈ Finset.range ranks, Real.exp (lse classes (fun j => x (rank*classes+j))))=
      lse (ranks*classes) x := by
  have each : ∀ rank∈Finset.range ranks,
      Real.exp (lse classes (fun j => x (rank*classes+j)))=expSum classes (fun j => x (rank*classes+j)) := by
    intro rank _
    exact Real.exp_log (expSum_positive classes _ positive)
  rw [Finset.sum_congr rfl each,rank_exp_partition]
  rfl

/-- A target contributes once precisely when it belongs to this source block. -/
theorem block_target (classes width start target : Nat) (value : ℝ) :
    (∑ i ∈ Finset.range width, if start+i<classes ∧ start+i=target then value else 0)=
      if start≤target ∧ target<min classes (start+width) then value else 0 := by
  by_cases inside : start≤target ∧ target<min classes (start+width)
  · rw [if_pos inside]
    have laneBound : target-start<width := by omega
    have address : start+(target-start)=target := by omega
    rw [Finset.sum_eq_single (target-start)]
    · have targetBound : target<classes := lt_of_lt_of_le inside.2 (Nat.min_le_left _ _)
      simp [address,targetBound]
    · intro i _ different
      have miss : start+i≠target := by omega
      simp [miss]
    · intro missing
      exact False.elim (missing (Finset.mem_range.mpr laneBound))
  · rw [if_neg inside]
    apply Finset.sum_eq_zero
    intro i hi
    have miss : ¬(start+i<classes ∧ start+i=target) := by
      have bound := Finset.mem_range.mp hi
      omega
    simp [miss]

theorem block_zero_sum (classes width : Nat) (covered : classes≤width) (f : Nat → ℝ) :
    blockSum classes width 0 f=∑ i∈Finset.range classes, f i := by
  simpa only [blockSum,Nat.zero_mul,Nat.zero_add] using sum_padding classes width covered f

theorem block_zero_lse (classes width : Nat) (covered : classes≤width) (x : Nat → ℝ) :
    blockLSE classes width 0 x=lse classes x := by
  rw [blockLSE,block_zero_sum classes width covered]
  rfl

/-- The padded blocks contribute the target exactly once, even at the last
partial block. The outside case is needed for non-owning distributed ranks. -/
theorem partition_target (classes width target : Nat) (positive : 0<width) (value : ℝ) :
    (∑ block∈Finset.range (blocks classes width),
      if block*width≤target ∧ target<min classes ((block+1)*width) then value else 0)=
      if target<classes then value else 0 := by
  have each : ∀ block,
      (if block*width≤target ∧ target<min classes ((block+1)*width) then value else 0)=
        blockSum classes width block (fun i => if i=target then value else 0) := by
    intro block
    rw [show (block+1)*width=block*width+width by ring]
    rw [← block_target classes width (block*width) target value]
    apply Finset.sum_congr rfl
    intro i _
    by_cases h : block*width+i<classes <;> by_cases ht : block*width+i=target <;>
      simp [blockSum,h,ht]
  simp_rw [each]
  rw [partition classes width positive]
  by_cases ht : target<classes
  · rw [if_pos ht,Finset.sum_eq_single target]
    · simp
    · intro i _ hi
      simp [hi]
    · simp [ht]
  · rw [if_neg ht]
    apply Finset.sum_eq_zero
    intro i hi
    have ne : i≠target := by have h := Finset.mem_range.mp hi; omega
    simp [ne]

#axiomsClean partition_target
#axiomsClean partition
#axiomsClean local_lse_composition
#axiomsClean rank_lse_composition
#axiomsClean block_target
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.Math

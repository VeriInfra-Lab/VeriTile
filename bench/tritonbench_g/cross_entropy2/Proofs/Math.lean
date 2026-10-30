import bench.tritonbench_g.cross_entropy2.Kernels
import bench.tritonbench_g.cross_entropy1.Proofs.Math

namespace VeriTile.Bench.Repairs.CrossEntropy2.Math
open scoped BigOperators
noncomputable section

/-- Mathematical target over every supplied column. Class metadata identifies
the optional target and the smoothing denominator; no unavailable remote logits
are inferred. Every int32 label has defined non-owning or ignored behavior. -/
def scaled (a : ℝ) (x : Nat → ℝ) (j : Nat) : ℝ := x j*a
def smoothing (s : ℝ) (enabled : Bool) : ℝ := if enabled then s else 0
def LSE (cols : Nat) (a : ℝ) (x : Nat → ℝ) : ℝ := VeriTile.Bench.Repairs.CrossEntropy1.Math.lse cols (scaled a x)
def targetTerm (cols : Nat) (start target : Int) (s a : ℝ) (x : Nat → ℝ) : ℝ :=
  if start≤target ∧ target<start+(cols : Int) then (1-s)*scaled a x (target-start).toNat else 0
def loss (cols total : Nat) (start target ignored : Int) (s a z : ℝ) (x : Nat → ℝ) : ℝ :=
  if target=ignored then 0 else LSE cols a x-targetTerm cols start target s a x-
    s*(∑ j∈Finset.range cols,scaled a x j)/total+z*LSE cols a x*LSE cols a x
def zLoss (cols : Nat) (target ignored : Int) (a z : ℝ) (x : Nat → ℝ) : ℝ :=
  if target=ignored then 0 else z*LSE cols a x*LSE cols a x
def gradient (cols total : Nat) (start target ignored : Int) (s a z upstream : ℝ)
    (x : Nat → ℝ) (j : Nat) : ℝ :=
  if target=ignored then 0 else (upstream*a)*
    ((1+2*z*LSE cols a x)*Real.exp (scaled a x j-LSE cols a x)-
      (if start+(j : Int)=target then 1-s else 0)-s/total)

/-- Standalone backward consumes the caller's supplied LSE tensor. The
forward/backward theorem instantiates this scalar with the proved forward LSE. -/
def gradientWithLSE (total : Nat) (start target ignored : Int) (s a z upstream L : ℝ)
    (x : Nat → ℝ) (j : Nat) : ℝ :=
  if target=ignored then 0 else (upstream*a)*
    ((1+2*z*L)*Real.exp (scaled a x j-L)-
      (if start+(j : Int)=target then 1-s else 0)-s/total)

def partialLoss (cols total width block : Nat) (start target ignored : Int)
    (s a : ℝ) (x : Nat → ℝ) : ℝ :=
  if target=ignored then 0 else
    -s*VeriTile.Bench.Repairs.CrossEntropy1.Math.blockSum cols width block (scaled a x)/total-
      if (block*width : Int)≤target-start ∧ target-start<(min cols ((block+1)*width) : Nat)
        then (1-s)*scaled a x (target-start).toNat else 0

theorem signed_target_partition (cols width : Nat) (positive : 0<width) (start target : Int) (v : ℝ) :
    (∑ b∈Finset.range (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols width),
      if (b*width : Int)≤target-start ∧ target-start<(min cols ((b+1)*width) : Nat) then v else 0)=
        if start≤target ∧ target<start+(cols : Int) then v else 0 := by
  by_cases nonneg : 0≤target-start
  · have cast : ((target-start).toNat : Int)=target-start := Int.toNat_of_nonneg nonneg
    have each : ∀ b : Nat, ((b*width : Int)≤target-start ∧ target-start<(min cols ((b+1)*width) : Nat)) ↔
        b*width≤(target-start).toNat ∧ (target-start).toNat<min cols ((b+1)*width) := by
      intro b
      rw [← cast]
      norm_cast
    simp_rw [each]
    rw [VeriTile.Bench.Repairs.CrossEntropy1.Math.partition_target cols width (target-start).toNat positive v]
    have rangeEq : (target-start).toNat<cols ↔ start≤target ∧ target<start+(cols : Int) := by omega
    simp only [rangeEq]
  · have outside : ¬(start≤target ∧ target<start+(cols : Int)) := by omega
    rw [if_neg outside]
    apply Finset.sum_eq_zero
    intro b _
    have miss : ¬((b*width : Int)≤target-start ∧ target-start<(min cols ((b+1)*width) : Nat)) := by
      have nonnegative : (0 : Int)≤(b*width : Int) := mul_nonneg (Int.natCast_nonneg b) (Int.natCast_nonneg width)
      omega
    exact if_neg miss

/-- Every padded block contribution is summed; this is a derived complete
reduction, with no whole-reduction numerical premise. -/
theorem partial_partition (cols total width : Nat) (positive : 0<width)
    (start target ignored : Int) (s a : ℝ) (x : Nat → ℝ) :
    (∑ b∈Finset.range (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols width),partialLoss cols total width b start target ignored s a x)=
      if target=ignored then 0 else
        -s*(∑ j∈Finset.range cols,scaled a x j)/total-targetTerm cols start target s a x := by
  by_cases hi : target=ignored
  · simp [partialLoss,hi]
  · simp only [partialLoss,hi,↓reduceIte,Finset.sum_sub_distrib,div_eq_mul_inv,← Finset.sum_mul,← Finset.mul_sum]
    rw [VeriTile.Bench.Repairs.CrossEntropy1.Math.partition cols width positive,signed_target_partition cols width positive]
    rfl

theorem complete_split (cols total width : Nat) (positive : 0<width)
    (start target ignored : Int) (s a z : ℝ) (x : Nat → ℝ) :
    (if target=ignored then 0 else
      (∑ b∈Finset.range (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols width),partialLoss cols total width b start target ignored s a x)+
        Real.log (∑ b∈Finset.range (VeriTile.Bench.Repairs.CrossEntropy1.Math.blocks cols width),
          Real.exp (VeriTile.Bench.Repairs.CrossEntropy1.Math.blockLSE cols width b (scaled a x)))+
        z*LSE cols a x*LSE cols a x)=loss cols total start target ignored s a z x := by
  rw [VeriTile.Bench.Repairs.CrossEntropy1.Math.local_lse_composition cols width positive,partial_partition cols total width positive]
  by_cases hi : target=ignored
  · simp [hi,loss]
  · simp only [hi,↓reduceIte,loss,LSE]
    ring

#axiomsClean partial_partition
#axiomsClean complete_split
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.Math

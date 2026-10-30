import bench.repairs.tritonbench_g.parallel_retention_attention.Proofs.Geometry

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention
noncomputable section
open scoped BigOperators

/-- Independent head decay, with its original -5 exponent. -/
def gamma (h : Nat) : ℝ := 1-(2:ℝ)^(-(5+(h:ℝ)))

theorem gamma_pos (h : Nat) : 0 < gamma h := by
  unfold gamma
  have hn : -(5+(h:ℝ)) < 0 := by linarith [Nat.cast_nonneg (α := ℝ) h]
  have hp : (2:ℝ)^(-(5+(h:ℝ))) < 1 := Real.rpow_lt_one_of_one_lt_of_neg (by norm_num) hn
  linarith

theorem logDecay_real (h : Nat) : logDecay realOps h=Real.log (gamma h)/Real.log 2 := by
  change Real.log (1-Real.exp ((-5-(h:ℝ)*1)*Real.log 2))/Real.log 2 = _
  have e : Real.exp ((-5-(h:ℝ)*1)*Real.log 2)=(2:ℝ)^(-(5+(h:ℝ))) := by
    rw [Real.rpow_def_of_pos (by norm_num : (0:ℝ)<2)]
    congr 1
    ring
  rw [e]
  rfl

theorem decayNat_real (h n : Nat) : decayNat realOps (logDecay realOps h) n=gamma h^n := by
  rw [logDecay_real]
  change Real.exp ((n:ℝ)*(Real.log (gamma h)/Real.log 2)*Real.log 2)=_
  have hlog : Real.log 2≠0 := ne_of_gt (Real.log_pos (by norm_num))
  have hex : (n:ℝ)*(Real.log (gamma h)/Real.log 2)*Real.log 2=(n:ℝ)*Real.log (gamma h) := by field_simp
  rw [hex,Real.exp_nat_mul,Real.exp_log (gamma_pos h)]

theorem blockDecay_real (h n : Nat) : blockDecay realOps (logDecay realOps h) n=gamma h^n := by
  have e : blockDecay realOps (logDecay realOps h) n=decayNat realOps (logDecay realOps h) n := by
    change Real.exp (logDecay realOps h*(n:ℝ)*Real.log 2)=Real.exp ((n:ℝ)*logDecay realOps h*Real.log 2)
    rw [mul_comm (logDecay realOps h)]
  rw [e,decayNat_real]

/-- Increasing-time source recurrence: multiply the actual live carry before
adding the next block. This is separate from its closed finite sum. -/
def leftScan (g : ℝ) (x : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => leftScan g x n*g+x n

theorem leftScan_sum (g : ℝ) (x : Nat → ℝ) (n : Nat) :
    leftScan g x n=∑ i∈Finset.range n,g^(n-1-i)*x i := by
  induction n with
  | zero => simp [leftScan]
  | succ n ih =>
    rw [leftScan,ih,Finset.sum_mul,Finset.sum_range_succ]
    simp only [Nat.add_sub_cancel,Nat.sub_self,pow_zero,one_mul]
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    have he : n-i=(n-1-i)+1 := by simp only [Finset.mem_range] at hi; omega
    rw [he,pow_succ]
    ring

/-- Decreasing-time source recurrence. Recursion evaluates the high-index tail
first and then multiplies its carry before adding this lower block. -/
def rightScan (g : ℝ) (x : Nat → ℝ) : Nat → ℝ
  | 0 => 0
  | n+1 => rightScan g (fun i => x (i+1)) n*g+x 0

theorem rightScan_sum (g : ℝ) (x : Nat → ℝ) (n : Nat) :
    rightScan g x n=∑ i∈Finset.range n,g^i*x i := by
  induction n generalizing x with
  | zero => simp [rightScan]
  | succ n ih =>
    rw [rightScan,ih,Finset.sum_mul,Finset.sum_range_succ']
    simp only [pow_zero,one_mul]
    congr 1
    apply Finset.sum_congr rfl
    intro i hi
    rw [pow_succ]
    ring

/-- Previous full BTS blocks followed by row rescaling. Padding and feature
sums are in x, so no recurrence identity is assumed at the call site. -/
theorem previous_blocks (g : ℝ) (x : Nat → ℝ) (J S r : Nat) (hS : 0<S) :
    leftScan (g^S) (fun j => ∑ u∈Finset.range S,g^(S-u)*x (j*S+u)) J*g^r=
      ∑ n∈Finset.range (J*S),g^(J*S+r-n)*x n := by
  rw [leftScan_sum,Finset.sum_mul,← sum_tiles J S]
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.mul_sum,Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro u hu
  simp only [Finset.mem_range] at hj hu
  rw [← pow_mul]
  have he : S*(J-1-j)+(S-u)+r=J*S+r-(j*S+u) := by
    have hj' : j+1≤J := by omega
    have hu' : u≤S := by omega
    have hs : J-1-j+j+1=J := by omega
    have ht : S-u+u=S := by omega
    have hb : j*S+u≤J*S+r := by nlinarith
    have hcancel := Nat.sub_add_cancel hb
    nlinarith
  calc
    g^(S*(J-1-j))*(g^(S-u)*x (j*S+u))*g^r =
        (g^(S*(J-1-j))*g^(S-u)*g^r)*x (j*S+u) := by ring
    _ = g^(J*S+r-(j*S+u))*x (j*S+u) := by rw [← pow_add,← pow_add,he]

/-- Later full BTS blocks evaluated from high to low. -/
theorem future_blocks (g : ℝ) (x : Nat → ℝ) (J S start extra : Nat) :
    rightScan (g^S) (fun j => ∑ u∈Finset.range S,g^u*x (start+j*S+u)) J*g^extra=
      ∑ n∈Finset.range (J*S),g^(n+extra)*x (start+n) := by
  rw [rightScan_sum,Finset.sum_mul,← sum_tiles J S]
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.mul_sum,Finset.sum_mul]
  apply Finset.sum_congr rfl
  intro u hu
  rw [← pow_mul]
  calc
    g^(S*j)*(g^u*x (start+j*S+u))*g^extra =
        (g^(S*j)*g^u*g^extra)*x (start+(j*S+u)) := by ring
    _ = g^(j*S+u+extra)*x (start+(j*S+u)) := by rw [← pow_add,← pow_add,Nat.mul_comm S j]

#axiomsClean gamma_pos
#axiomsClean decayNat_real
#axiomsClean leftScan_sum
#axiomsClean rightScan_sum
#axiomsClean previous_blocks
#axiomsClean future_blocks
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention

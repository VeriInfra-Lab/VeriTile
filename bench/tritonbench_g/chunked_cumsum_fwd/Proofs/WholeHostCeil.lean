import bench.tritonbench_g.chunked_cumsum_fwd.Proofs.WholeHost
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.HostBoundary
noncomputable section
set_option maxHeartbeats 2000000

/-- The ordinary correctly-rounded binary64 division contract on nonnegative
signed32 Python integers. Integral quotients are exactly representable. For
nonintegral quotients the standard normal-number relative error is 2^-53.
This host contract is separate from all pending GPU numerical primitives;
it supplies neither the ceiling nor a launch shape nor a kernel result. -/
structure Binary64Division where
  quotient : Nat → Nat → ℝ
  integral : ∀ n d, n < 2^31 → 0 < d → d < 2^31 → n % d = 0 →
    quotient n d = (n / d : Nat)
  relative : ∀ n d, n < 2^31 → 0 < d → d < 2^31 →
    |quotient n d - (n : ℝ)/(d : ℝ)| ≤ ((n : ℝ)/(d : ℝ))/2^53

theorem ceilDiv_remainder (n d : Nat) (hd : 0 < d) :
    Whole.ceilDiv n d = n/d + if n%d = 0 then 0 else 1 := by
  have hr := Nat.mod_lt n hd
  have he : n = d*(n/d)+n%d := (Nat.div_add_mod n d).symm
  unfold Whole.ceilDiv
  split_ifs with hz
  · have hs : n+d-1 = d*(n/d)+(d-1) := by omega
    rw [hs,Nat.mul_add_div hd,Nat.div_eq_of_lt (show d-1 < d by omega)]
  · have hs : n+d-1 = d*(n/d+1)+(n%d-1) := by
      rw [Nat.mul_add,Nat.mul_one]
      omega
    rw [hs,Nat.mul_add_div hd,Nat.div_eq_of_lt (show n%d-1 < d by omega)]

/-- Python's rounded quotient has the exact mathematical ceiling throughout
the signed32 host-size domain, even when chunk size is not a power of two. -/
theorem binary64_ceil_exact (fp : Binary64Division) (n d : Nat)
    (hn : n < 2^31) (hd : 0 < d) (hd32 : d < 2^31) :
    Nat.ceil (fp.quotient n d) = Whole.ceilDiv n d := by
  rw [ceilDiv_remainder n d hd]
  by_cases hz : n%d = 0
  · rw [if_pos hz,fp.integral n d hn hd hd32 hz,Nat.ceil_natCast,Nat.add_zero]
  · rw [if_neg hz]
    have dr : (0 : ℝ) < d := by exact_mod_cast hd
    have nr : (n : ℝ) < 2^31 := by exact_mod_cast hn
    have remLow : (1 : ℝ) ≤ ((n%d : Nat) : ℝ) := by exact_mod_cast (show 1 ≤ n%d by omega)
    have remHigh : ((n%d : Nat) : ℝ)+1 ≤ d := by exact_mod_cast (show n%d+1 ≤ d from Nat.mod_lt n hd)
    have equation : (n : ℝ) = (n/d : Nat)*(d : ℝ)+((n%d : Nat) : ℝ) := by
      exact_mod_cast (show n = n/d*d+n%d by simpa [Nat.mul_comm] using (Nat.div_add_mod n d).symm)
    have err := fp.relative n d hn hd hd32
    have small : ((n : ℝ)/(d : ℝ))/2^53 < 1/(d : ℝ) := by
      apply (div_lt_div_iff₀ (by positivity : (0 : ℝ) < 2^53) dr).2
      rw [div_mul_cancel₀ _ (ne_of_gt dr)]
      norm_num at nr ⊢
      linarith
    have delta := lt_of_le_of_lt err small
    have range := abs_lt.mp delta
    have lo : (n/d : Nat) < fp.quotient n d := by
      have e : (n : ℝ)/(d : ℝ) = (n/d : Nat)+((n%d : Nat) : ℝ)/(d : ℝ) := by
        apply (div_eq_iff (ne_of_gt dr)).2
        rw [add_mul,div_mul_cancel₀ _ (ne_of_gt dr)]
        exact equation
      rw [e] at range
      have := (div_le_div_iff_of_pos_right dr).2 remLow
      linarith
    have hi : fp.quotient n d ≤ (n/d : Nat)+1 := by
      have e : (n : ℝ)/(d : ℝ)+1/(d : ℝ) ≤ (n/d : Nat)+1 := by
        apply (le_of_mul_le_mul_right ?_ dr)
        rw [add_mul,div_mul_cancel₀ _ (ne_of_gt dr),div_mul_cancel₀ _ (ne_of_gt dr)]
        nlinarith
      linarith
    apply (Nat.ceil_eq_iff (Nat.succ_ne_zero (n/d))).2
    simpa only [Nat.succ_eq_add_one,Nat.add_sub_cancel,Nat.cast_add,Nat.cast_one] using And.intro lo hi

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkedCumsumFwd.HostBoundary

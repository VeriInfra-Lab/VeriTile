import bench.repairs.tritonbench_g.int8_matmul_quantization.Proofs.KernelNativeModel
import Mathlib.Tactic

/-! Exact integer and scalar arithmetic for the independently admitted
constant-input counterexample. This is diagnostic evidence, not a substitute
for a missing source-to-result correctness proof. Actual admission, native
specialization and emitted instruction selection are recorded beside it. -/
namespace VeriTile.Bench.Repairs.Int8MatmulQuantization.ScalarCastCounterexample
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 32000

def maxInt32 : Nat := 2147483647

def carry (term : Nat) : Nat → Nat
  | 0 => 0
  | n+1 => min maxInt32 (carry term n + term)

theorem saturating_step (limit before term : Nat) :
    min limit (min limit before + term) = min limit (before + term) := by
  omega

theorem carry_sum (term steps : Nat) : carry term steps = min maxInt32 (steps*term) := by
  induction steps with
  | zero => simp [carry]
  | succ n ih => rw [carry,ih,saturating_step,Nat.succ_mul]

/-- Every output lane sees 8192 K32 atoms. Even the smallest code permitted
by the documented producer divide error already saturates signed INT32. -/
theorem composed_carries :
    carry (32*126*128) 8192 = maxInt32 ∧ carry (32*127*128) 8192 = maxInt32 := by
  rw [carry_sum,carry_sum]
  decide

/-- No saturation is reached in either half-length SPLIT_K=2 recurrence. -/
theorem split_control_carries :
    carry (32*126*128) 4096 = 2113929216 ∧
    carry (32*127*128) 4096 = 2130706432 := by
  rw [carry_sum,carry_sum]
  decide

/-- `div.full.f32` has at most two ulps of error. The first interval below
is a wider enclosure around 127/127; the second is two ulps near -127.
This lemma proves the arithmetic consequence of those documented local
bounds. It makes no claim that the native divide is exactly IEEE division. -/
theorem producer_quotient_bounds (scale quotient : ℝ)
    (hs0 : 1-1/(2^21 : ℝ) ≤ scale) (hs1 : scale ≤ 1+1/(2^21 : ℝ))
    (hq0 : -127/scale-1/(2^16 : ℝ) ≤ quotient)
    (hq1 : quotient ≤ -127/scale+1/(2^16 : ℝ)) :
    -128 < quotient ∧ quotient < -126 := by
  have hs : 0 < scale := by norm_num at hs0; linarith
  have eq : (-127/scale)*scale = (-127 : ℝ) := div_mul_cancel₀ _ (ne_of_gt hs)
  have lower : (-128 : ℝ) < -127/scale-1/(2^16 : ℝ) := by
    norm_num at hs0 hs1 ⊢
    nlinarith
  have upper : -127/scale+1/(2^16 : ℝ) < (-126 : ℝ) := by
    norm_num at hs0 hs1 ⊢
    nlinarith
  exact ⟨lt_of_lt_of_le lower hq0,lt_of_le_of_lt hq1 upper⟩

/-- The producer's whole live-scale interval is strictly inside the binary16
RNE cell of 1 (different gaps above and below that power of two). -/
theorem stored_scale_cell (scale : ℝ)
    (lo : 1-1/(2^21 : ℝ) ≤ scale) (hi : scale ≤ 1+1/(2^21 : ℝ)) :
    1-1/(2^12 : ℝ) < scale ∧ scale < 1+1/(2^11 : ℝ) := by
  constructor <;> norm_num at * <;> linarith

def convert (n : Nat) : FP.Value .fp32 := FP.round .fp32 .ieeeValue (n : Rat)
def scaledOutput (n : Nat) := Int8DequantMatmul.narrow16
  (FP.mul .ieeeValue (FP.mul .ieeeValue (convert n) (FP.Value.ofBits .fp32 0x3f800000))
    (FP.Value.ofBits .fp32 0x30000000))

/-- INT32_MAX rounds to 2^31 in FP32, then multiplication by the exact scale
1 and 2^-31 gives the exactly representable binary16 value 1. -/
theorem saturated_output :
    (convert maxInt32).bits = BitVec.ofNat 32 0x4f000000 ∧
    (scaledOutput maxInt32).bits = BitVec.ofNat 16 0x3c00 := by
  decide +kernel

theorem mathematical_target :
    ((-127 : ℝ)*(-128)*262144)*1*(1/(2^31 : ℝ)) = 127/64 ∧
    (127/64 : ℝ) ≠ 1 := by norm_num

#axiomsClean carry_sum
#axiomsClean composed_carries
#axiomsClean split_control_carries
#axiomsClean producer_quotient_bounds
#axiomsClean stored_scale_cell
#axiomsClean saturated_output
#axiomsClean mathematical_target
end
end VeriTile.Bench.Repairs.Int8MatmulQuantization.ScalarCastCounterexample

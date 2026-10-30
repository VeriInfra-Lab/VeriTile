import VeriTile.Triton.Random.Proofs.RoundExpansion
import VeriTile.Triton.Random.Proofs.DyadicStep

namespace VeriTile.Triton.Random.RoundBridge

set_option maxRecDepth 10000
set_option maxHeartbeats 2000000

theorem positive_normal_raw (e f : Nat) (hemax : e < 255) (hf : f < 2^23) :
    (FP.Value.ofBits .fp32 (e*2^23+f)).raw = e*2^23+f := by
  simp only [FP.Value.raw, FP.Value.ofBits, BitVec.toNat_ofNat]
  apply Nat.mod_eq_of_lt
  change e*2^23+f < 2^32
  omega

theorem positive_normal_exponent (e f : Nat) (hemax : e < 255) (hf : f < 2^23) :
    (FP.Value.ofBits .fp32 (e*2^23+f)).exponent = e := by
  change ((FP.Value.ofBits .fp32 (e*2^23+f)).raw / 2^23) % 256 = e
  rw [positive_normal_raw e f hemax hf]
  omega

theorem positive_normal_fraction (e f : Nat) (hemax : e < 255) (hf : f < 2^23) :
    (FP.Value.ofBits .fp32 (e*2^23+f)).fraction = f := by
  change (FP.Value.ofBits .fp32 (e*2^23+f)).raw % 2^23 = f
  rw [positive_normal_raw e f hemax hf]
  omega

theorem positive_normal_sign (e f : Nat) (hemax : e < 255) (hf : f < 2^23) :
    (FP.Value.ofBits .fp32 (e*2^23+f)).sign = false := by
  change ((FP.Value.ofBits .fp32 (e*2^23+f)).raw / 2^31 != 0) = false
  rw [positive_normal_raw e f hemax hf]
  have hz : (e*2^23+f) / 2^31 = 0 := by omega
  rw [hz]
  rfl

/-- Decode a positive normal bit pattern, before any rounding argument. -/
theorem decode_positive_normal (e f : Nat)
    (he : 0 < e) (hemax : e < 255) (hf : f < 2^23) :
    (FP.Value.ofBits .fp32 (e*2^23+f)).decode =
      .finite false (Rat.ofInt ((2^23+f:Nat):Int)*FP.pow2 ((e:Int)-150)) := by
  unfold FP.Value.decode
  dsimp only
  rw [positive_normal_exponent e f hemax hf,
    positive_normal_fraction e f hemax hf, positive_normal_sign e f hemax hf]
  have hei : (e == 255) = false := by simp only [beq_eq_false_iff_ne];omega
  have hez : (e == 0) = false := by simp only [beq_eq_false_iff_ne];omega
  rw [hei, hez]
  have hi : (e:Int)-127-23 = (e:Int)-150 := by omega
  change FP.Decoded.finite false (Rat.ofInt ((2^23+f:Nat):Int) * FP.pow2 ((e:Int)-127-23)) = _
  rw [hi]

/-- Packing a normalized significand decodes to its exact scale product. -/
theorem decode_scaled_significand (step : Int) (n : Nat)
    (hstep : -149 ≤ step) (hmax : step ≤ 104)
    (hn : 2^23 ≤ n) (hnmax : n < 2^24) :
    (FP.Value.ofBits .fp32 ((step+150).toNat*2^23+n-2^23)).decode =
      .finite false ((n:Rat)*FP.pow2 step) := by
  have he : 0 < (step+150).toNat := by omega
  have hemax : (step+150).toNat < 255 := by omega
  have hf : n-2^23 < 2^23 := by omega
  have hraw : (step+150).toNat*2^23+n-2^23 =
      (step+150).toNat*2^23+(n-2^23) := by omega
  rw [hraw, decode_positive_normal _ _ he hemax hf]
  have heq : ((step+150).toNat:Int)-150 = step := by omega
  rw [heq]
  have hsig : 2^23+(n-2^23) = n := by omega
  rw [hsig]
  rfl

end VeriTile.Triton.Random.RoundBridge

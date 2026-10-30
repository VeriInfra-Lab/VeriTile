import VeriTile.Triton.Float.CastExact.Fields
import VeriTile.Triton.Random.Proofs.DyadicStep

namespace VeriTile.Triton.FP.CastExact

private theorem scaled_pow2 (n : Nat) (e : Int) :
    (n * 65536 : Rat) * pow2 (e - 16) = (n : Rat) * pow2 e := by
  rw [Random.RoundBridge.pow2_eq_zpow, Random.RoundBridge.pow2_eq_zpow,
    zpow_sub₀ (by norm_num : (2 : Rat) ≠ 0)]
  norm_num
  ring

theorem widenBits_isNaN (x : Value .bf16) : (widenBits x).isNaN = x.isNaN := by
  apply Bool.eq_iff_iff.mpr
  simp [Value.isNaN, widenBits_exponent, widenBits_fraction, bne_iff_ne]

theorem widenBits_decode_finite (x : Value .bf16) (s : Bool) (m : Rat)
    (h : x.decode = .finite s m) : (widenBits x).decode = .finite s m := by
  have he : x.exponent ≠ 255 := by
    intro he
    simp only [Value.decode, he, beq_self_eq_true, ite_true] at h
    split at h <;> contradiction
  simp only [Value.decode, widenBits_exponent, widenBits_fraction, widenBits_sign,
    Format.fractionBits] at h ⊢
  simp only [beq_iff_eq, he, ↓reduceIte] at h ⊢
  rw [← h]
  congr 1
  by_cases hz : x.exponent = 0
  · simpa [hz, Rat.ofInt_eq_cast] using scaled_pow2 x.fraction (-133)
  · simp only [hz, ↓reduceIte, Nat.cast_ofNat]
    have hn : (2 ^ 23 + x.fraction * 65536 : Nat) = (128 + x.fraction) * 65536 := by omega
    rw [hn]
    have he' : (x.exponent : Int) - 127 - 23 = ((x.exponent : Int) - 127 - 7) - 16 := by omega
    rw [he']
    norm_num only [Nat.reducePow]
    simpa only [Nat.cast_mul, Nat.cast_ofNat, Int.cast_natCast, Int.cast_mul,
      Rat.ofInt_eq_cast] using scaled_pow2 (128 + x.fraction) ((x.exponent : Int) - 127 - 7)

end VeriTile.Triton.FP.CastExact

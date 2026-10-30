import VeriTile.Triton.Float.CastExact.Fields

namespace VeriTile.Triton.FP.CastExact

/-- A bounded finite exponent and fraction cannot encode a NaN. -/
theorem packed_not_nan (f : Format) (s : Bool) (e n : Nat)
    (he : e < 255) (hn : n < 2 ^ f.fractionBits) :
    (Value.ofBits f ((if s then 2 ^ (f.fractionBits + 8) else 0) +
      e * 2 ^ f.fractionBits + n)).isNaN = false := by
  cases f <;> cases s <;>
    simp only [Value.isNaN, Value.exponent, Value.fraction, Value.raw,
      Value.ofBits, BitVec.toNat_ofNat, Format.width, Format.fractionBits,
      Bool.false_eq_true, ↓reduceIte] at * <;>
    norm_num at * <;> omega

theorem subnormal_pack_not_nan (f : Format) (s : Bool) (n : Nat)
    (hn : n < 2 ^ f.fractionBits) :
    (Value.ofBits f ((if s then 2 ^ (f.fractionBits + 8) else 0) + n)).isNaN = false := by
  simpa only [Nat.zero_mul, Nat.add_zero] using packed_not_nan f s 0 n (by decide) hn

theorem normal_pack_not_nan (f : Format) (s : Bool) (e : Int) (n : Nat)
    (he : -126 ≤ e ∧ e ≤ 127)
    (hn : 2 ^ f.fractionBits ≤ n ∧ n < 2 ^ (f.fractionBits + 1)) :
    (Value.ofBits f ((if s then 2 ^ (f.fractionBits + 8) else 0) +
      (e + 127).toNat * 2 ^ f.fractionBits + n - 2 ^ f.fractionBits)).isNaN = false := by
  have he' : (e + 127).toNat < 255 := by omega
  have hn' : n - 2 ^ f.fractionBits < 2 ^ f.fractionBits := by
    cases f <;> norm_num [Format.fractionBits] at hn ⊢ <;> omega
  have h := packed_not_nan f s (e + 127).toNat (n - 2 ^ f.fractionBits) he' hn'
  convert h using 1
  congr 2
  omega

end VeriTile.Triton.FP.CastExact

import VeriTile.Triton.Float.BitValue
import Mathlib.Tactic

/-! Bit-field identities used by the exact cast laws. No rounding rule is
assumed here, and no enumeration of the FP32 word space is used. -/
namespace VeriTile.Triton.FP.CastExact

theorem raw_lt {f : Format} (x : Value f) : x.raw < 2 ^ f.width :=
  x.bits.isLt

theorem exponent_lt {f : Format} (x : Value f) : x.exponent < 256 := by
  exact Nat.mod_lt _ (by decide)

theorem fraction_lt {f : Format} (x : Value f) : x.fraction < 2 ^ f.fractionBits := by
  exact Nat.mod_lt _ (by positivity)

theorem raw_decomposition {f : Format} (x : Value f) :
    x.raw = (if x.sign then 2 ^ (f.fractionBits + 8) else 0) +
      x.exponent * 2 ^ f.fractionBits + x.fraction := by
  have bound := raw_lt x
  cases f <;>
    simp only [Value.sign, Value.exponent, Value.fraction, Format.fractionBits,
      Format.width, bne_iff_ne] at * <;>
    norm_num at * <;>
    split_ifs <;> omega

theorem ofBits_fields {f : Format} (x : Value f) :
    Value.ofBits f ((if x.sign then 2 ^ (f.fractionBits + 8) else 0) +
      x.exponent * 2 ^ f.fractionBits + x.fraction) = x := by
  rw [← raw_decomposition x]
  cases x
  simp [Value.ofBits, Value.raw]

theorem eq_infinity_of_decode {f : Format} (x : Value f) (s : Bool)
    (h : x.decode = .infinity s) : x = Value.infinity f s := by
  simp only [Value.decode] at h
  split at h
  · rename_i he
    split at h
    · rename_i hf
      have he' : x.exponent = 255 := by simpa using he
      have hf' : x.fraction = 0 := by simpa using hf
      have hs : x.sign = s := Decoded.infinity.inj h
      change x = Value.ofBits f ((if s then 2 ^ (f.fractionBits + 8) else 0) +
        255 * 2 ^ f.fractionBits + 0)
      simpa only [he', hf', hs] using (ofBits_fields x).symm
    · contradiction
  · contradiction

@[simp] theorem decode_infinity (f : Format) (s : Bool) :
    (Value.infinity f s).decode = .infinity s := by
  cases f <;> cases s <;> decide

@[simp] theorem decode_qNaN (f : Format) :
    (Value.qNaN f).decode = .nan false false (2 ^ (f.fractionBits - 1)) := by
  cases f <;> decide

@[simp] theorem isNaN_qNaN (f : Format) : (Value.qNaN f).isNaN = true := by
  cases f <;> decide

/-- The exact word-level embedding of BF16 into FP32. For NaNs this preserves
the payload; the software `cast` will subsequently canonicalize it. -/
def widenBits (x : Value .bf16) : Value .fp32 := Value.ofBits .fp32 (x.raw * 65536)

theorem widenBits_raw (x : Value .bf16) : (widenBits x).raw = x.raw * 65536 := by
  have bound := raw_lt x
  norm_num [Format.width, Format.fractionBits] at bound
  change (x.raw * 65536) % 4294967296 = x.raw * 65536
  exact Nat.mod_eq_of_lt (by omega)

theorem widenBits_sign (x : Value .bf16) : (widenBits x).sign = x.sign := by
  simp only [Value.sign, widenBits_raw, Format.fractionBits]
  norm_num
  congr 1
  omega

theorem widenBits_exponent (x : Value .bf16) : (widenBits x).exponent = x.exponent := by
  simp only [Value.exponent, widenBits_raw, Format.fractionBits]
  norm_num
  congr 1
  omega

theorem widenBits_fraction (x : Value .bf16) :
    (widenBits x).fraction = x.fraction * 65536 := by
  simp only [Value.fraction, widenBits_raw, Format.fractionBits]
  norm_num
  omega

end VeriTile.Triton.FP.CastExact

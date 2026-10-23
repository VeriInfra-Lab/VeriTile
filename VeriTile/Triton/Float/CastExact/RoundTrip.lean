import VeriTile.Triton.Float.CastExact.SameFormat
import VeriTile.Triton.Float.CastExact.Widen

namespace VeriTile.Triton.FP.CastExact

/-- BF16-to-FP32 conversion inserts sixteen zero fraction bits for every
finite BF16 value, including both signed zeros and all subnormal words. -/
theorem cast_widen_of_decode_finite (x : Value .bf16) (s : Bool) (m : Rat)
    (h : x.decode = .finite s m) : cast .fp32 .ieeeValue x = widenBits x := by
  simpa only [cast, prepare_ieee, h] using
    BitValueLaws.round_of_decode_finite (widenBits x) s m
      (widenBits_decode_finite x s m h)

theorem cast_widen_of_finite (x : Value .bf16) (h : x.toRat?.isSome = true) :
    cast .fp32 .ieeeValue x = widenBits x := by
  cases hd : x.decode with
  | finite s m => exact cast_widen_of_decode_finite x s m hd
  | infinity s => simp [Value.toRat?, hd] at h
  | nan s signaling payload => simp [Value.toRat?, hd] at h

theorem bf16_widen_return_of_decode_finite (x : Value .bf16) (s : Bool) (m : Rat)
    (h : x.decode = .finite s m) :
    cast .bf16 .ieeeValue (cast .fp32 .ieeeValue x) = x := by
  rw [cast_widen_of_decode_finite x s m h]
  simpa only [cast, prepare_ieee, widenBits_decode_finite x s m h] using
    BitValueLaws.round_of_decode_finite x s m h

/-- The exact widening roundtrip preserves all non-NaN BF16 words. NaN payloads
are intentionally excluded because `cast` canonicalizes NaNs. -/
theorem bf16_widen_return (x : Value .bf16) (h : x.isNaN = false) :
    cast .bf16 .ieeeValue (cast .fp32 .ieeeValue x) = x := by
  cases hd : x.decode with
  | finite s m => exact bf16_widen_return_of_decode_finite x s m hd
  | infinity s =>
    rw [eq_infinity_of_decode x s hd]
    simp
  | nan s signaling payload =>
    rw [isNaN_of_decode_nan x s signaling payload hd] at h
    contradiction

/-- On arbitrary BF16 words, the roundtrip is exactly NaN canonicalization. -/
theorem bf16_widen_return_normalized (x : Value .bf16) :
    cast .bf16 .ieeeValue (cast .fp32 .ieeeValue x) =
      if x.isNaN then Value.qNaN .bf16 else x := by
  cases hd : x.decode with
  | finite s m =>
    rw [isNaN_of_decode_finite x s m hd]
    exact bf16_widen_return_of_decode_finite x s m hd
  | infinity s =>
    rw [isNaN_of_decode_infinity x s hd]
    exact bf16_widen_return x (isNaN_of_decode_infinity x s hd)
  | nan s signaling payload =>
    simp [cast, hd, isNaN_of_decode_nan x s signaling payload hd]

theorem bf16_widen_return_eq_cast_same (x : Value .bf16) :
    cast .bf16 .ieeeValue (cast .fp32 .ieeeValue x) = cast .bf16 .ieeeValue x := by
  rw [bf16_widen_return_normalized, cast_same]

#axiomsClean cast_widen_of_decode_finite
#axiomsClean cast_widen_of_finite
#axiomsClean bf16_widen_return
#axiomsClean bf16_widen_return_normalized

end VeriTile.Triton.FP.CastExact

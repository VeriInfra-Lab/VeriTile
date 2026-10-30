import VeriTile.Triton.Float.CastExact.RoundTrip
import VeriTile.Triton.Float.BitValueRoundClassification

namespace VeriTile.Triton.FP.CastExact

/-- A cast may produce a NaN only in the model's canonical form. This follows
from the result classification of the actual rational rounding operation. -/
theorem cast_nan_canonical {src dst : Format} (x : Value src)
    (h : (cast dst .ieeeValue x).isNaN = true) :
    cast dst .ieeeValue x = Value.qNaN dst := by
  cases hd : x.decode with
  | finite s m =>
    simp only [cast, prepare_ieee, hd,
      BitValueRoundClassification.round_not_nan] at h
    contradiction
  | infinity s =>
    have hn : (Value.infinity dst s).isNaN = false := by
      cases dst <;> cases s <;> decide
    simp only [cast, prepare_ieee, hd, hn] at h
    contradiction
  | nan s signaling payload => simp only [cast, prepare_ieee, hd]

/-- Repeating a conversion to its already selected format preserves every
result word. No finiteness or input-NaN-payload restriction is needed. -/
theorem cast_idempotent {src : Format} (dst : Format) (x : Value src) :
    cast dst .ieeeValue (cast dst .ieeeValue x) = cast dst .ieeeValue x := by
  rw [cast_same]
  split
  · exact (cast_nan_canonical x ‹(cast dst .ieeeValue x).isNaN = true›).symm
  · rfl

/-- Narrowing to BF16, widening exactly, and narrowing again is idempotent for
all input words. Arbitrary NaN payloads are canonicalized by the first cast. -/
theorem bf16_round_idempotent {src : Format} (x : Value src) :
    cast .bf16 .ieeeValue (cast .fp32 .ieeeValue (cast .bf16 .ieeeValue x)) =
      cast .bf16 .ieeeValue x := by
  rw [bf16_widen_return_eq_cast_same, cast_idempotent]

/-- The concrete ROUND-IDEM law for both supported destination formats,
including both signed zeros, all subnormals, infinities, and NaNs. -/
theorem round_idempotent {src : Format} (dst : Format) (x : Value src) :
    cast dst .ieeeValue (cast .fp32 .ieeeValue (cast dst .ieeeValue x)) =
      cast dst .ieeeValue x := by
  cases dst with
  | bf16 => exact bf16_round_idempotent x
  | fp32 => rw [cast_idempotent, cast_idempotent]

#axiomsClean cast_nan_canonical
#axiomsClean cast_idempotent
#axiomsClean bf16_round_idempotent
#axiomsClean round_idempotent

end VeriTile.Triton.FP.CastExact

import VeriTile.Triton.Float.BitValueLaws

namespace VeriTile.Triton.FP.CastExact

@[simp] theorem prepare_ieee {f : Format} (x : Value f) :
    prepare .ieeeValue x = x := by
  simp [prepare, Config.ieeeValue]

/-- Same-format conversion preserves every finite word, including signed zeros
and all subnormals. -/
theorem cast_same_of_decode_finite {f : Format} (x : Value f) (s : Bool) (m : Rat)
    (h : x.decode = .finite s m) : cast f .ieeeValue x = x := by
  simpa only [cast, prepare_ieee, h] using
    BitValueLaws.round_of_decode_finite x s m h

theorem cast_same_of_finite {f : Format} (x : Value f)
    (h : x.toRat?.isSome = true) : cast f .ieeeValue x = x := by
  cases hd : x.decode with
  | finite s m => exact cast_same_of_decode_finite x s m hd
  | infinity s => simp [Value.toRat?, hd] at h
  | nan s signaling payload => simp [Value.toRat?, hd] at h

theorem isNaN_of_decode_nan {f : Format} (x : Value f) (s signaling : Bool) (payload : Nat)
    (h : x.decode = .nan s signaling payload) : x.isNaN = true := by
  simp only [Value.decode] at h
  split at h
  · rename_i he
    split at h
    · contradiction
    · rename_i hf
      simp only [Value.isNaN, he, Bool.true_and, bne_iff_ne]
      simpa only [beq_iff_eq] using hf
  · contradiction

theorem isNaN_of_decode_finite {f : Format} (x : Value f) (s : Bool) (m : Rat)
    (h : x.decode = .finite s m) : x.isNaN = false := by
  simp only [Value.decode] at h
  split at h
  · split at h <;> contradiction
  · rename_i he
    simp [Value.isNaN, he]

theorem isNaN_of_decode_infinity {f : Format} (x : Value f) (s : Bool)
    (h : x.decode = .infinity s) : x.isNaN = false := by
  rw [eq_infinity_of_decode x s h]
  cases f <;> cases s <;> decide

@[simp] theorem cast_infinity (src dst : Format) (s : Bool) :
    cast dst .ieeeValue (Value.infinity src s) = Value.infinity dst s := by
  simp [cast]

@[simp] theorem cast_qNaN (src dst : Format) :
    cast dst .ieeeValue (Value.qNaN src) = Value.qNaN dst := by
  simp [cast]

/-- Same-format conversion canonicalizes NaNs and preserves every other word. -/
theorem cast_same (f : Format) (x : Value f) :
    cast f .ieeeValue x = if x.isNaN then Value.qNaN f else x := by
  cases hd : x.decode with
  | finite s m =>
    rw [isNaN_of_decode_finite x s m hd]
    exact cast_same_of_decode_finite x s m hd
  | infinity s =>
    rw [eq_infinity_of_decode x s hd]
    have hn : (Value.infinity f s).isNaN = false := by cases f <;> cases s <;> decide
    simp [hn]
  | nan s signaling payload =>
    simp [cast, hd, isNaN_of_decode_nan x s signaling payload hd]

theorem cast_same_of_not_nan {f : Format} (x : Value f) (h : x.isNaN = false) :
    cast f .ieeeValue x = x := by
  simp [cast_same, h]

theorem cast_same_idempotent {f : Format} (x : Value f) :
    cast f .ieeeValue (cast f .ieeeValue x) = cast f .ieeeValue x := by
  rw [cast_same f x]
  split
  · exact cast_qNaN f f
  · exact cast_same_of_not_nan x (by simpa using ‹¬ x.isNaN = true›)

#axiomsClean cast_same_of_decode_finite
#axiomsClean cast_same_of_finite
#axiomsClean cast_same
#axiomsClean cast_same_idempotent

end VeriTile.Triton.FP.CastExact

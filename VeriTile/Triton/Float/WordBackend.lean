import VeriTile.Triton.Float.Countermodel
import VeriTile.Triton.Float.CastExact.RoundTrip
import VeriTile.Triton.Float.ScalarOpsLaws

/-!
A software backend for mixed BF16/FP32 kernels whose carrier is one FP32 word.

The strict FP32 theory (`Guarded.fp32Theory`) is certified on `strictFP32`,
where every carrier element is an FP32 word. `Guarded.Models` asks a backend
to validate those certificates on every state of its own carrier. A carrier
with elements that are not FP32 words cannot do that. A pair
`Value .bf16 × Value .fp32`, for example, contains pairs whose components
disagree; no Triton tensor holds such a value, and an FP32 primitive applied
to one returns a consistent pair, so `max x x` and `x` differ.

Every BF16 value is exactly an FP32 value (`CastExact.widenBits` appends
sixteen zero fraction bits and keeps NaN payloads). A BF16 tile therefore
needs no second component: this backend stores it as its widened word. The
tile dtype in the program, not the carrier, records that the value is BF16.

`mixedFP32 rest` is literally `strictFP32 (bf16Words rest)`. It therefore
models the strict theory by the existing certificate transport, and its FP32
primitives are the strict ones. The BF16 layer adds only what a mixed kernel
needs: the two conversions, the four rounded BF16 arithmetic operations and
the literals 0 and 1. Other operations stay in `rest`.

A word that is not the widening of a BF16 value may still sit in a BF16
position, since the carrier does not record dtypes. `narrowWord` rounds such
a word to BF16. This is a totality choice outside the reachable states of a
well-typed kernel; no strict FP32 law depends on it.
-/
namespace VeriTile.Triton.FP.Software
open Structural Guarded CastExact

/-- Read an FP32 word in a BF16 position. A widened BF16 word returns its
exact source, including NaN payloads; any other word is rounded to BF16. -/
def narrowWord (x : FP.Value .fp32) : FP.Value .bf16 :=
  if x.raw % 65536 = 0 then FP.Value.ofBits .bf16 (x.raw / 65536)
  else FP.cast .bf16 .ieeeValue x

/-- Widening is injective on all BF16 words: `narrowWord` recovers the source. -/
@[simp] theorem narrowWord_widenBits (x : FP.Value .bf16) : narrowWord (widenBits x) = x := by
  unfold narrowWord
  rw [widenBits_raw, if_pos (Nat.mul_mod_left _ _),
    Nat.mul_div_cancel _ (by decide : 0 < 65536)]
  cases x with
  | mk bits =>
    refine congrArg FP.Value.mk (BitVec.eq_of_toNat_eq ?_)
    simp only [FP.Value.raw, BitVec.toNat_ofNat]
    exact Nat.mod_eq_of_lt bits.isLt

theorem widenBits_injective : Function.Injective widenBits :=
  fun a b h => by simpa using congrArg narrowWord h

/-- The BF16 layer underneath the strict FP32 primitives. The dtype tag of the
operation selects it; an FP32 precision context never reaches this layer. -/
noncomputable def bf16Words (rest : Algebra (FP.Value .fp32)) : Algebra (FP.Value .fp32) :=
  { rest with
    literal := fun p d x =>
      if d = .bf16 then
        if x = 0 then widenBits (FP.Value.zero .bf16)
        else if x = 1 then widenBits (ScalarOpsLaws.one .bf16)
        else rest.literal p d x
      else rest.literal p d x
    binary := fun p d op a b =>
      if d = .bf16 then
        match op with
        | .add => widenBits (FP.add .ieeeValue (narrowWord a) (narrowWord b))
        | .sub => widenBits (FP.sub .ieeeValue (narrowWord a) (narrowWord b))
        | .mul => widenBits (FP.mul .ieeeValue (narrowWord a) (narrowWord b))
        | .div => widenBits (FP.div .ieeeValue (narrowWord a) (narrowWord b))
        | _ => rest.binary p d op a b
      else rest.binary p d op a b
    cast := fun p src dst x =>
      if src = dst then x
      else if (src = .real ∨ src = .fp32) ∧ (dst = .real ∨ dst = .fp32) then x
      else if src = .bf16 ∧ (dst = .real ∨ dst = .fp32) then
        FP.cast .fp32 .ieeeValue (narrowWord x)
      else if (src = .real ∨ src = .fp32) ∧ dst = .bf16 then
        widenBits (FP.cast .bf16 .ieeeValue x)
      else rest.cast p src dst x }

/-- Strict FP32 primitives over one-word registers, with BF16 tiles stored as
their exact widened words. -/
noncomputable def mixedFP32 (rest : Algebra (FP.Value .fp32)) : Algebra (FP.Value .fp32) :=
  strictFP32 (bf16Words rest)

theorem mixedFP32_eq (rest : Algebra (FP.Value .fp32)) :
    mixedFP32 rest = strictFP32 (bf16Words rest) := rfl

private theorem not_fp32Context_bf16 (p : Option ComputeDType) :
    fp32Context p .bf16 = false := by
  rcases p with _ | p
  · rfl
  · cases p <;> rfl

/-- A same-dtype cast returns its operand. -/
@[simp] theorem mixedFP32_cast_same (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (x : FP.Value .fp32) :
    (mixedFP32 rest).cast p d d x = x := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words]

/-- Widening a stored BF16 value is the shared rounded conversion. -/
theorem mixedFP32_cast_widen (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (x : FP.Value .bf16) :
    (mixedFP32 rest).cast p .bf16 .fp32 (widenBits x) = FP.cast .fp32 .ieeeValue x := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words]

theorem mixedFP32_cast_widen_real (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (x : FP.Value .bf16) :
    (mixedFP32 rest).cast p .bf16 .real (widenBits x) = FP.cast .fp32 .ieeeValue x := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words]

/-- A finite BF16 value widens to the word that already stores it. -/
theorem mixedFP32_cast_widen_of_finite (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (x : FP.Value .bf16) (finite : x.toRat?.isSome = true) :
    (mixedFP32 rest).cast p .bf16 .fp32 (widenBits x) = widenBits x := by
  rw [mixedFP32_cast_widen, cast_widen_of_finite x finite]

/-- Narrowing rounds once to BF16 and stores the exact widened result. -/
theorem mixedFP32_cast_narrow (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (x : FP.Value .fp32) :
    (mixedFP32 rest).cast p .fp32 .bf16 x = widenBits (FP.cast .bf16 .ieeeValue x) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words]

theorem mixedFP32_cast_narrow_real (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (x : FP.Value .fp32) :
    (mixedFP32 rest).cast p .real .bf16 x = widenBits (FP.cast .bf16 .ieeeValue x) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words]

/-- Widening a stored non-NaN BF16 value and narrowing it again returns the
stored word. NaN payloads are excluded because `FP.cast` canonicalizes NaNs. -/
theorem mixedFP32_widen_narrow_return (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (x : FP.Value .bf16) (notNaN : x.isNaN = false) :
    (mixedFP32 rest).cast p .fp32 .bf16 ((mixedFP32 rest).cast p .bf16 .fp32 (widenBits x)) =
      widenBits x := by
  rw [mixedFP32_cast_widen, mixedFP32_cast_narrow, bf16_widen_return x notNaN]

/-- BF16 arithmetic rounds in BF16, whatever precision context is ambient. -/
theorem mixedFP32_bf16_add (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (a b : FP.Value .bf16) :
    (mixedFP32 rest).binary p .bf16 .add (widenBits a) (widenBits b) =
      widenBits (FP.add .ieeeValue a b) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words, not_fp32Context_bf16]

theorem mixedFP32_bf16_sub (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (a b : FP.Value .bf16) :
    (mixedFP32 rest).binary p .bf16 .sub (widenBits a) (widenBits b) =
      widenBits (FP.sub .ieeeValue a b) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words, not_fp32Context_bf16]

theorem mixedFP32_bf16_mul (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (a b : FP.Value .bf16) :
    (mixedFP32 rest).binary p .bf16 .mul (widenBits a) (widenBits b) =
      widenBits (FP.mul .ieeeValue a b) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words, not_fp32Context_bf16]

theorem mixedFP32_bf16_div (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (a b : FP.Value .bf16) :
    (mixedFP32 rest).binary p .bf16 .div (widenBits a) (widenBits b) =
      widenBits (FP.div .ieeeValue a b) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words, not_fp32Context_bf16]

theorem mixedFP32_bf16_zero (rest : Algebra (FP.Value .fp32)) (p : Option ComputeDType) :
    (mixedFP32 rest).literal p .bf16 0 = widenBits (FP.Value.zero .bf16) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words, not_fp32Context_bf16]

theorem mixedFP32_bf16_one (rest : Algebra (FP.Value .fp32)) (p : Option ComputeDType) :
    (mixedFP32 rest).literal p .bf16 1 = widenBits (ScalarOpsLaws.one .bf16) := by
  simp [mixedFP32, strictFP32, scalarFP32, bf16Words, not_fp32Context_bf16]

/-- The FP32 primitives are the strict ones; the BF16 layer does not shadow them. -/
theorem mixedFP32_add (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (a b : FP.Value .fp32)
    (h : fp32Context p d = true) :
    (mixedFP32 rest).binary p d .add a b = FP.add .ieeeValue a b :=
  strictFP32_add _ p d a b h

theorem mixedFP32_mul (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (a b : FP.Value .fp32)
    (h : fp32Context p d = true) :
    (mixedFP32 rest).binary p d .mul a b = FP.mul .ieeeValue a b :=
  strictFP32_mul _ p d a b h

theorem mixedFP32_sub (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (a b : FP.Value .fp32)
    (h : fp32Context p d = true) :
    (mixedFP32 rest).binary p d .sub a b = FP.sub .ieeeValue a b :=
  strictFP32_sub _ p d a b h

/-- Distinct FP32 words stay distinct in the mixed backend. -/
theorem mixedFP32_faithful (rest : Algebra (FP.Value .fp32)) : FaithfulWords (mixedFP32 rest) :=
  strict_faithful _

/-- The mixed backend interprets the strict theory with no experimental rule. -/
theorem mixedFP32_models_empty (rest : Algebra (FP.Value .fp32)) :
    Models [] (mixedFP32 rest) strictDomain :=
  strict_models_empty _

/-- On any operand domain inside the strict one, the mixed backend models a
rule table as soon as each listed atom holds there. The strict certificates
need no further argument because the carrier holds only FP32 words. -/
theorem mixedFP32_models (rest : Algebra (FP.Value .fp32))
    (D : Domain (FP.Value .fp32))
    (domain : ∀ kind value, D kind value → strictDomain kind value)
    (R : Spec.Assumptions GuardedFragment)
    (atoms : ∀ e ∈ R, List.Forall₂ (FragmentSound (mixedFP32 rest) D) e.rule.lhs e.rule.rhs) :
    Models R (mixedFP32 rest) D :=
  models_of_atoms _ D R (fun _ x _ => strict_word _ x) atoms
    (fun lhs rhs certificate => strict_software_interpretation _ D domain lhs rhs certificate)

#axiomsClean narrowWord_widenBits
#axiomsClean widenBits_injective
#axiomsClean mixedFP32_cast_same
#axiomsClean mixedFP32_cast_widen
#axiomsClean mixedFP32_cast_widen_real
#axiomsClean mixedFP32_cast_widen_of_finite
#axiomsClean mixedFP32_cast_narrow
#axiomsClean mixedFP32_cast_narrow_real
#axiomsClean mixedFP32_widen_narrow_return
#axiomsClean mixedFP32_bf16_add
#axiomsClean mixedFP32_bf16_sub
#axiomsClean mixedFP32_bf16_mul
#axiomsClean mixedFP32_bf16_div
#axiomsClean mixedFP32_bf16_zero
#axiomsClean mixedFP32_bf16_one
#axiomsClean mixedFP32_add
#axiomsClean mixedFP32_mul
#axiomsClean mixedFP32_sub
#axiomsClean mixedFP32_faithful
#axiomsClean mixedFP32_models_empty
#axiomsClean mixedFP32_models
end VeriTile.Triton.FP.Software

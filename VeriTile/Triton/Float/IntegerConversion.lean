import VeriTile.Triton.Float.ScalarBackend
import Mathlib.Data.Rat.Floor
import VeriTile.Meta.StatementAudit

/-!
Defined-domain, toward-zero numeric conversions from concrete binary values.
The input is decoded from its bits to an exact rational; no Real evaluator is
used. NaN, infinity, width zero and integer overflow fail. This is an explicit
partial software profile. Native lowering, overflow and saturation behavior
must be bound separately; this module does not choose those behaviors for
Triton/CUDA. The adapter accepts only an actual fp32 source context.
-/
namespace VeriTile.Triton.FP.IntegerConversion

/-- Exact quotient computation: ceil on negative values, floor otherwise. -/
def truncate (q : Rat) : Int := if q < 0 then -(-q).floor else q.floor

/-- Independent interval specification for rounding toward zero. -/
def TowardZero (q : Rat) (z : Int) : Prop :=
  if q < 0 then (z : Rat) - 1 < q ∧ q ≤ z
  else (z : Rat) ≤ q ∧ q < (z : Rat) + 1

theorem truncate_spec (q : Rat) : TowardZero q (truncate q) := by
  unfold TowardZero truncate
  split
  · exact Int.ceil_eq_iff.mp rfl
  · exact Int.floor_eq_iff.mp rfl

theorem truncate_unique (q : Rat) (z : Int) : TowardZero q z ↔ truncate q = z := by
  unfold TowardZero truncate
  split
  · change ((z : Rat) - 1 < q ∧ q ≤ z) ↔ ⌈q⌉ = z
    exact Int.ceil_eq_iff.symm
  · change ((z : Rat) ≤ q ∧ q < (z : Rat) + 1) ↔ ⌊q⌋ = z
    exact Int.floor_eq_iff.symm

def SignedRange (width : Nat) (z : Int) : Prop :=
  0 < width ∧ -(2 ^ (width - 1) : Int) ≤ z ∧ z < 2 ^ (width - 1)

def UnsignedRange (width : Nat) (z : Int) : Prop :=
  0 < width ∧ 0 ≤ z ∧ z < (2 ^ width : Int)

instance (width : Nat) (z : Int) : Decidable (SignedRange width z) := inferInstanceAs
  (Decidable (0 < width ∧ -(2 ^ (width - 1) : Int) ≤ z ∧ z < 2 ^ (width - 1)))
instance (width : Nat) (z : Int) : Decidable (UnsignedRange width z) := inferInstanceAs
  (Decidable (0 < width ∧ 0 ≤ z ∧ z < (2 ^ width : Int)))

def toSigned {f : Format} (width : Nat) (x : FP.Value f) : Option Int := do
  let q ← x.toRat?
  let z := truncate q
  if SignedRange width z then some z else none

def toUnsigned {f : Format} (width : Nat) (x : FP.Value f) : Option Nat := do
  let q ← x.toRat?
  let z := truncate q
  if UnsignedRange width z then some z.toNat else none

/-- Every successful execution has an exact finite interpretation, rounds
within the independent interval specification and fits the signed word. -/
theorem toSigned_spec {f : Format} (width : Nat) (x : FP.Value f) (z : Int) :
    toSigned width x = some z ↔
      ∃ q, x.toRat? = some q ∧ TowardZero q z ∧ SignedRange width z := by
  cases hx : x.toRat? with
  | none => simp [toSigned, hx]
  | some q =>
      simp only [toSigned, hx, bind, Option.bind_some, Option.some.injEq]
      constructor
      · split
        · intro h
          cases h
          exact ⟨q, rfl, truncate_spec q, by assumption⟩
        · simp
      · rintro ⟨q', hq, hz, hr⟩
        cases hq
        have ht := (truncate_unique q z).mp hz
        simp [ht, hr]

/-- Unsigned conversion also checks the lower bound after truncation, so
negative fractions strictly between -1 and 0 are well-defined zero. -/
theorem toUnsigned_spec {f : Format} (width : Nat) (x : FP.Value f) (z : Nat) :
    toUnsigned width x = some z ↔
      ∃ q, x.toRat? = some q ∧ TowardZero q (z : Int) ∧ UnsignedRange width (z : Int) := by
  cases hx : x.toRat? with
  | none => simp [toUnsigned, hx]
  | some q =>
      simp only [toUnsigned, hx, bind, Option.bind_some, Option.some.injEq]
      constructor
      · split
        · intro h
          cases h
          have hr : UnsignedRange width (truncate q) := by assumption
          have he : ((truncate q).toNat : Int) = truncate q := Int.toNat_of_nonneg hr.2.1
          refine ⟨q, rfl, ?_, ?_⟩
          · simpa only [he] using truncate_spec q
          · simpa only [he] using hr
        · simp
      · rintro ⟨q', hq, hz, hr⟩
        cases hq
        have ht := (truncate_unique q (z : Int)).mp hz
        simp [ht, hr]

/-- A format tag alone suffices for an ordinary typed fp32 expression. An
erased Real carrier additionally requires the explicit fp32 compute tag. -/
def fp32Source : Option IndexComputePrecision → FloatDType → Bool
  | none, .fp32 | some .fp32, .fp32 | some .fp32, .real => true
  | _, _ => false

noncomputable def fp32Backend (rest : Structural.Algebra (FP.Value .fp32)) :
    Structural.Algebra (FP.Value .fp32) :=
  { rest with
    floatToInt := fun p src width =>
      if fp32Source p src then some (toSigned width) else none
    floatToNat := fun p src width =>
      if fp32Source p src then some (toUnsigned width) else none }

@[simp] theorem fp32Backend_signed (rest : Structural.Algebra (FP.Value .fp32))
    (p : Option IndexComputePrecision) (src : FloatDType) (width : Nat)
    (h : fp32Source p src = true) :
    (fp32Backend rest).floatToInt p src width = some (toSigned width) := by
  simp [fp32Backend, h]

@[simp] theorem fp32Backend_unsigned (rest : Structural.Algebra (FP.Value .fp32))
    (p : Option IndexComputePrecision) (src : FloatDType) (width : Nat)
    (h : fp32Source p src = true) :
    (fp32Backend rest).floatToNat p src width = some (toUnsigned width) := by
  simp [fp32Backend, h]

-- Kernel evaluation of representative bit patterns, including all failure classes.
example : toSigned 8 (FP.Value.ofBits .fp32 0xbff00000) = some (-1) := by decide +kernel
example : toUnsigned 8 (FP.Value.ofBits .fp32 0x437fe000) = some 255 := by decide +kernel
example : toUnsigned 8 (FP.Value.ofBits .fp32 0xbf000000) = some 0 := by decide +kernel
example : toSigned 8 (FP.Value.ofBits .fp32 0x43000000) = none := by decide +kernel
example : toUnsigned 8 (FP.Value.ofBits .fp32 0x43800000) = none := by decide +kernel
example : toUnsigned 8 (FP.Value.ofBits .fp32 0xbf800000) = none := by decide +kernel
example : toSigned 32 (FP.Value.infinity .fp32) = none := by decide +kernel
example : toUnsigned 32 (FP.Value.qNaN .fp32) = none := by decide +kernel
example : toSigned 8 (FP.Value.zero .fp32 true) = some 0 := by decide +kernel
example : toSigned 0 (FP.Value.zero .fp32) = none := by decide +kernel

#axiomsClean truncate_spec
#axiomsClean truncate_unique
#axiomsClean toSigned_spec
#axiomsClean toUnsigned_spec
#axiomsClean fp32Backend_signed
#axiomsClean fp32Backend_unsigned
end VeriTile.Triton.FP.IntegerConversion

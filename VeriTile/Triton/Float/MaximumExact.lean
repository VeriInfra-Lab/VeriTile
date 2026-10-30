import VeriTile.Triton.Float.BitValue
import VeriTile.Meta.StatementAudit
import Mathlib.Data.Nat.Lattice
import Mathlib.Tactic.SplitIfs

/-! Exact laws for the existing maxnum profile, at FP32 and BF16 precision.

The model below is extracted without changing its branches from
`bench/tritonbench_g/max_reduction/Proofs/Native.lean`:
one NaN is ignored, two NaNs return the selected target word, and a tie between
zero words uses bitwise AND, selecting +0 unless both inputs are -0. The same
zero policy is explicit in `quantize_copy_kv/Proofs/FPMaxProfile.lean` and
`chunked_cumsum_fwd/Proofs/KernelNumerics.lean`. Triton's default
`PropagateNan.NONE` maximum lowers to maxnumf; this file proves properties of
that explicitly selected software profile, not hardware conformance.

Finite means exponent < 255: both zero signs and every subnormal are included.
There is no input/result flush. An arbitrary comparison/select implementation
does not inherit these bitwise laws merely by having the same Real projection.
-/
namespace VeriTile.Triton.FP.MaximumExact
namespace Generic

abbrev Word (f : Format) := BitVec f.width

variable {f : Format}

def magnitude (x : Word f) : Nat := x.toNat % 2^(f.fractionBits + 8)
def sign (x : Word f) : Bool := decide (2^(f.fractionBits + 8) ≤ x.toNat)
def nan (x : Word f) : Bool := decide ((255 * 2^f.fractionBits) < magnitude x)
def zero (x : Word f) : Bool := decide (magnitude x = 0)

def greater (x y : Word f) : Bool :=
  !nan x && !nan y && !(zero x && zero y) &&
    (if sign x = sign y then
      if sign x then decide (magnitude x < magnitude y)
      else decide (magnitude y < magnitude x)
    else !sign x)

structure Target (f : Format) where
  canonicalNaN : Word f
  isNaN : nan canonicalNaN = true
  quiet : canonicalNaN.toNat / 2^(f.fractionBits - 1) % 2 = 1

/-- The existing software profile's positive quiet-NaN payload. -/
def canonicalTarget (f : Format) : Target f :=
  ⟨(Value.qNaN f).bits, by cases f <;> decide +kernel,
    by cases f <;> decide +kernel⟩

def maximum (target : Target f) (x y : Word f) : Word f :=
  if nan x then (if nan y then target.canonicalNaN else y)
  else if nan y then x
  else if greater x y then x
  else if greater y x then y
  else if zero x && zero y then x &&& y
  else y

def maximumValue (x y : Value f) : Value f :=
  ⟨maximum (canonicalTarget f) x.bits y.bits⟩

def finite (x : Word f) : Prop := magnitude x < (255 * 2^f.fractionBits)

/-- This is the usual finite-word domain, not a restriction excluding signed
zeros or subnormals. -/
theorem finite_iff_exponent (x : Word f) :
    finite x ↔ (⟨x⟩ : Value f).exponent < 255 := by
  have bound := x.isLt
  cases f <;>
    simp only [finite, magnitude, Value.exponent, Value.raw, Format.fractionBits,
      Format.width] at * <;> omega

theorem finite_iff_toRat (x : Word f) :
    finite x ↔ ((⟨x⟩ : Value f).toRat?).isSome = true := by
  rw [finite_iff_exponent]
  have bound : (⟨x⟩ : Value f).exponent < 256 :=
    Nat.mod_lt _ (by decide)
  by_cases h : (⟨x⟩ : Value f).exponent = 255
  · by_cases hf : (⟨x⟩ : Value f).fraction = 0 <;>
      simp [Value.toRat?, Value.decode, h, hf]
  · have lt : (⟨x⟩ : Value f).exponent < 255 := by omega
    simp [Value.toRat?, Value.decode, h, lt]

/-- An injective ordering key for raw words. Negative magnitudes are reversed;
the two zero encodings remain distinct, with -0 immediately below +0. This is
a proof device, not the definition of the maximum primitive. -/
def rank (x : Word f) : Nat :=
  if 2^(f.fractionBits + 8) ≤ x.toNat then 2^f.width - 1 - x.toNat else 2^(f.fractionBits + 8) + x.toNat

theorem rank_injective : Function.Injective (rank (f := f)) := by
  intro x y h
  have hx := x.isLt
  have hy := y.isLt
  apply BitVec.eq_of_toNat_eq
  dsimp only [rank] at h
  cases f <;> simp only [Format.width, Format.fractionBits] at * <;>
    split_ifs at h <;> omega

theorem finite_not_nan (x : Word f) (h : finite x) : nan x = false := by
  simp only [finite] at h
  simp [nan, Nat.not_lt_of_ge (Nat.le_of_lt h)]

private theorem zero_cases (x : Word f) (h : zero x = true) :
    x = 0 ∨ x = BitVec.ofNat f.width (2^(f.fractionBits + 8)) := by
  have bound := x.isLt
  have hmod : x.toNat % 2^(f.fractionBits + 8) = 0 := by simpa [zero, magnitude] using h
  have parts : x.toNat = 0 ∨ x.toNat = 2^(f.fractionBits + 8) := by
    cases f <;> simp only [Format.width, Format.fractionBits] at * <;> omega
  rcases parts with h | h
  · left; exact BitVec.eq_of_toNat_eq h
  · right; apply BitVec.eq_of_toNat_eq
    cases f <;> simpa [Format.width, Format.fractionBits] using h

private theorem greater_rank (x y : Word f) (hx : nan x = false) (hy : nan y = false)
    (hz : (zero x && zero y) = false) :
    greater x y = decide (rank y < rank x) := by
  have bx := x.isLt
  have by_ := y.isLt
  simp only [greater, hx, hy, hz, Bool.not_false, Bool.true_and]
  cases f
  · dsimp only [Word, Format.width, Format.fractionBits] at *
    by_cases sx : 32768 ≤ x.toNat <;> by_cases sy : 32768 ≤ y.toNat <;>
      simp [sign, rank, magnitude, Format.width, Format.fractionBits] at * <;>
        split_ifs <;> simp_all <;> (try simp only [← decide_not, decide_eq_decide]) <;> omega
  · dsimp only [Word, Format.width, Format.fractionBits] at *
    by_cases sx : 2147483648 ≤ x.toNat <;> by_cases sy : 2147483648 ≤ y.toNat <;>
      simp [sign, rank, magnitude, Format.width, Format.fractionBits] at * <;>
        split_ifs <;> simp_all <;> (try simp only [← decide_not, decide_eq_decide]) <;> omega

theorem maximum_eq_rank (target : Target f) (x y : Word f)
    (hx : nan x = false) (hy : nan y = false) :
    maximum target x y = if rank x ≤ rank y then y else x := by
  by_cases hz : (zero x && zero y) = true
  · have zx : zero x = true := (Bool.and_eq_true_iff.mp hz).1
    have zy : zero y = true := (Bool.and_eq_true_iff.mp hz).2
    rcases zero_cases x zx with rfl | rfl <;>
      rcases zero_cases y zy with rfl | rfl <;>
      cases f <;> rfl
  · have hz' : (zero x && zero y) = false := Bool.eq_false_iff.mpr hz
    have hzy : (zero y && zero x) = false := by simpa [Bool.and_comm] using hz'
    simp only [maximum, hx, hy, Bool.false_eq_true, ↓reduceIte,
      greater_rank x y hx hy hz', greater_rank y x hy hx hzy, hz', decide_eq_true_eq]
    split_ifs <;> first | rfl | omega

theorem maximum_selects (target : Target f) (x y : Word f)
    (hx : nan x = false) (hy : nan y = false) :
    maximum target x y = x ∨ maximum target x y = y := by
  rw [maximum_eq_rank target x y hx hy]
  split <;> simp

theorem rank_maximum (target : Target f) (x y : Word f)
    (hx : nan x = false) (hy : nan y = false) :
    rank (maximum target x y) = max (rank x) (rank y) := by
  rw [maximum_eq_rank target x y hx hy]
  split_ifs with h
  · exact (max_eq_right h).symm
  · exact (max_eq_left (Nat.le_of_not_ge h)).symm

theorem maximum_not_nan (target : Target f) (x y : Word f)
    (hx : nan x = false) (hy : nan y = false) :
    nan (maximum target x y) = false := by
  rcases maximum_selects target x y hx hy with h | h <;> simp_all

theorem maximum_finite (target : Target f) (x y : Word f) (hx : finite x) (hy : finite y) :
    finite (maximum target x y) := by
  rcases maximum_selects target x y (finite_not_nan x hx) (finite_not_nan y hy) with h | h <;>
    simp_all

/-- MAX-COMMUTE, including both signs of zero and all subnormals. -/
theorem max_commute (target : Target f) (x y : Word f) (hx : finite x) (hy : finite y) :
    maximum target x y = maximum target y x := by
  apply rank_injective
  rw [rank_maximum target x y (finite_not_nan x hx) (finite_not_nan y hy),
    rank_maximum target y x (finite_not_nan y hy) (finite_not_nan x hx), max_comm]

/-- MAX-ASSOC, with no changed floating computation or rounding. -/
theorem max_assoc (target : Target f) (x y z : Word f)
    (hx : finite x) (hy : finite y) (hz : finite z) :
    maximum target (maximum target x y) z = maximum target x (maximum target y z) := by
  have nx := finite_not_nan x hx
  have ny := finite_not_nan y hy
  have nz := finite_not_nan z hz
  apply rank_injective
  rw [rank_maximum target _ z (maximum_not_nan target x y nx ny) nz,
    rank_maximum target x _ nx (maximum_not_nan target y z ny nz),
    rank_maximum target x y nx ny, rank_maximum target y z ny nz, _root_.max_assoc]

/-- MAX-IDEM preserves the actual input word, including -0. -/
theorem max_idem (target : Target f) (x : Word f) (hx : finite x) :
    maximum target x x = x := by
  apply rank_injective
  rw [rank_maximum target x x (finite_not_nan x hx) (finite_not_nan x hx), max_self]

def negativeInfinity (f : Format) : Word f :=
  BitVec.ofNat f.width (511 * 2^f.fractionBits)

/-- MAX-NEG-INF uses -infinity only as the designated literal; the sampled
operand remains finite and can have either zero sign. -/
theorem max_neg_inf (target : Target f) (x : Word f) (hx : finite x) :
    maximum target (negativeInfinity f) x = x := by
  have ni : nan (negativeInfinity f) = false := by cases f <;> decide +kernel
  rw [maximum_eq_rank target (negativeInfinity f) x ni (finite_not_nan x hx)]
  have bound := x.isLt
  have hmag := hx
  have hr : rank (negativeInfinity f) ≤ rank x := by
    rw [show rank (negativeInfinity f) = 2^f.fractionBits - 1 from by
      cases f <;> decide +kernel]
    dsimp only [rank]
    cases f <;> simp only [finite, magnitude, Format.width, Format.fractionBits] at * <;>
      split_ifs <;> omega
  exact if_pos hr

/-- Zero-sign boundary values are part of the proved domain, not samples used
as a substitute for the general theorems. -/
theorem zero_boundaries (target : Target f) :
    let p : Word f := 0
    let n : Word f := BitVec.ofNat f.width (2^(f.fractionBits + 8))
    finite p ∧ finite n ∧ maximum target p n = p ∧ maximum target n p = p ∧
      maximum target n n = n := by
  cases f <;> dsimp only
  all_goals
    refine ⟨by unfold finite magnitude; decide +kernel,
      by unfold finite magnitude; decide +kernel, ?_, ?_, ?_⟩ <;> rfl

/-- Typed-carrier forms use the same finite guard as the interpreter. -/
private theorem value_ext {x y : Value f} (h : x.bits = y.bits) : x = y := by
  cases x
  cases y
  cases h
  rfl

theorem maximumValue_commute (x y : Value f)
    (hx : x.toRat?.isSome = true) (hy : y.toRat?.isSome = true) :
    maximumValue x y = maximumValue y x := by
  apply value_ext
  exact max_commute (canonicalTarget f) x.bits y.bits
    ((finite_iff_toRat x.bits).mpr hx) ((finite_iff_toRat y.bits).mpr hy)

theorem maximumValue_assoc (x y z : Value f)
    (hx : x.toRat?.isSome = true) (hy : y.toRat?.isSome = true)
    (hz : z.toRat?.isSome = true) :
    maximumValue (maximumValue x y) z = maximumValue x (maximumValue y z) := by
  apply value_ext
  exact max_assoc (canonicalTarget f) x.bits y.bits z.bits
    ((finite_iff_toRat x.bits).mpr hx) ((finite_iff_toRat y.bits).mpr hy)
    ((finite_iff_toRat z.bits).mpr hz)

theorem maximumValue_idem (x : Value f) (hx : x.toRat?.isSome = true) :
    maximumValue x x = x := by
  apply value_ext
  exact max_idem (canonicalTarget f) x.bits ((finite_iff_toRat x.bits).mpr hx)

theorem maximumValue_neg_inf (x : Value f) (hx : x.toRat?.isSome = true) :
    maximumValue (Value.infinity f true) x = x := by
  apply value_ext
  change maximum (canonicalTarget f) (Value.infinity f true).bits x.bits = x.bits
  rw [show (Value.infinity f true).bits = negativeInfinity f from by cases f <;> rfl]
  exact max_neg_inf (canonicalTarget f) x.bits ((finite_iff_toRat x.bits).mpr hx)

#axiomsClean finite_iff_toRat
#axiomsClean maximum_finite
#axiomsClean max_commute
#axiomsClean max_assoc
#axiomsClean max_idem
#axiomsClean max_neg_inf
#axiomsClean zero_boundaries
#axiomsClean maximumValue_commute
#axiomsClean maximumValue_assoc
#axiomsClean maximumValue_idem
#axiomsClean maximumValue_neg_inf

end Generic

/-- FP32 compatibility names for the extracted model. -/
abbrev Word := Generic.Word .fp32
abbrev Target := Generic.Target .fp32
abbrev canonicalTarget := Generic.canonicalTarget .fp32
abbrev magnitude := @Generic.magnitude .fp32
abbrev sign := @Generic.sign .fp32
abbrev nan := @Generic.nan .fp32
abbrev zero := @Generic.zero .fp32
abbrev greater := @Generic.greater .fp32
abbrev maximum := @Generic.maximum .fp32
abbrev maximumValue := @Generic.maximumValue .fp32
abbrev finite := @Generic.finite .fp32
abbrev rank := @Generic.rank .fp32
abbrev negativeInfinity := Generic.negativeInfinity .fp32
abbrev finite_iff_exponent := @Generic.finite_iff_exponent .fp32
abbrev finite_iff_toRat := @Generic.finite_iff_toRat .fp32
abbrev maximum_finite := @Generic.maximum_finite .fp32
abbrev max_commute := @Generic.max_commute .fp32
abbrev max_assoc := @Generic.max_assoc .fp32
abbrev max_idem := @Generic.max_idem .fp32
abbrev max_neg_inf := @Generic.max_neg_inf .fp32
abbrev maximumValue_commute := @Generic.maximumValue_commute .fp32
abbrev maximumValue_assoc := @Generic.maximumValue_assoc .fp32
abbrev maximumValue_idem := @Generic.maximumValue_idem .fp32
abbrev maximumValue_neg_inf := @Generic.maximumValue_neg_inf .fp32

end VeriTile.Triton.FP.MaximumExact

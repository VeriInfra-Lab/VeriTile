import VeriTile.Triton.Float.BitValue
import VeriTile.Meta.StatementAudit
import Init.Data.BitVec.Lemmas

/-! Task-local binary64 words. This complete bit decoder deliberately does not
change Core.Float64Bits.decodeRat, whose domain is finite normal values only.
Packing is exact field encoding, not rounding from a rational number. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.Binary64Words
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

abbrev Word := BitVec 64

structure Fields where
  sign : BitVec 1
  exponent : BitVec 11
  fraction : BitVec 52
  deriving DecidableEq, Repr

def fields (x : Word) : Fields :=
  ⟨x.extractLsb' 63 1, x.extractLsb' 52 11, x.extractLsb' 0 52⟩

def encode (f : Fields) : Word := f.sign ++ (f.exponent ++ f.fraction)

theorem fields_encode (f : Fields) : fields (encode f) = f := by
  cases f with
  | mk s e m =>
    change Fields.mk ((s ++ (e ++ m)).extractLsb' 63 1)
      ((s ++ (e ++ m)).extractLsb' 52 11)
      ((s ++ (e ++ m)).extractLsb' 0 52) = Fields.mk s e m
    congr 1
    · exact @BitVec.extractLsb'_append_eq_left 1 63 s (e ++ m)
    · rw [@BitVec.extractLsb'_append_eq_of_add_le 1 63 s (e ++ m) 52 11 (by decide)]
      exact @BitVec.extractLsb'_append_eq_left 11 52 e m
    · rw [@BitVec.extractLsb'_append_eq_of_add_le 1 63 s (e ++ m) 0 52 (by decide)]
      exact @BitVec.extractLsb'_append_eq_right 11 52 e m

theorem encode_fields (x : Word) : encode (fields x) = x := by
  unfold encode fields
  rw [BitVec.extractLsb'_append_extractLsb'_eq_extractLsb' (by decide : 52 = 0 + 52)]
  exact BitVec.extractLsb'_append_extractLsb'

theorem fields_injective (x y : Word) (h : fields x = fields y) : x = y := by
  rw [← encode_fields x, ← encode_fields y, h]

def negative (x : Word) : Bool := (fields x).sign.msb
def exponent (x : Word) : Nat := (fields x).exponent.toNat
def fraction (x : Word) : Nat := (fields x).fraction.toNat

/-- Full binary64 decoder. Both zero signs, subnormals, infinities and every
NaN payload are retained; the top fraction bit distinguishes quiet NaNs. -/
def decode (x : Word) : FP.Decoded :=
  let e := exponent x
  let m := fraction x
  if e = 2047 then
    if m = 0 then .infinity (negative x)
    else .nan (negative x) (m < 2^51) m
  else
    let significand := if e = 0 then m else 2^52 + m
    let scale : Int := if e = 0 then -1074 else (e : Int) - 1075
    .finite (negative x) ((significand : Rat) * FP.pow2 scale)

def toRat? (x : Word) : Option Rat :=
  match decode x with
  | .finite sign magnitude => some (if sign then -magnitude else magnitude)
  | _ => none

theorem finite_projection_iff (x : Word) :
    (toRat? x).isSome = true ↔ exponent x ≠ 2047 := by
  by_cases he : exponent x = 2047 <;> by_cases hm : fraction x = 0 <;>
    simp [toRat?, decode, he, hm]

theorem special_decode (x : Word) (h : exponent x = 2047) :
    decode x = if fraction x = 0 then .infinity (negative x)
      else .nan (negative x) (fraction x < 2^51) (fraction x) := by
  simp [decode, h]

theorem subnormal_or_zero_decode (x : Word) (h : exponent x = 0) :
    decode x = .finite (negative x) ((fraction x : Rat) * FP.pow2 (-1074)) := by
  simp [decode, h]

/-- Kernel-reduced boundaries exercise all classes and preserve signed zeros
and the complete NaN fraction, including the quiet/signaling distinction. -/
theorem decode_boundaries :
    decode (0x0000000000000000#64) = .finite false 0 ∧
    decode (0x8000000000000000#64) = .finite true 0 ∧
    decode (0x0000000000000001#64) = .finite false (FP.pow2 (-1074)) ∧
    decode (0x8000000000000001#64) = .finite true (FP.pow2 (-1074)) ∧
    decode (0x000fffffffffffff#64) = .finite false (((2^52-1 : Nat) : Rat) * FP.pow2 (-1074)) ∧
    decode (0x0010000000000000#64) = .finite false (FP.pow2 (-1022)) ∧
    decode (0x3ff0000000000000#64) = .finite false 1 ∧
    decode (0x7fefffffffffffff#64) = .finite false (((2^53-1 : Nat) : Rat) * FP.pow2 971) ∧
    decode (0x7ff0000000000000#64) = .infinity false ∧
    decode (0xfff0000000000000#64) = .infinity true ∧
    decode (0x7ff8000000000123#64) = .nan false false (2^51+0x123) ∧
    decode (0xfff0000000000123#64) = .nan true true 0x123 := by
  decide +kernel

#axiomsClean fields_encode
#axiomsClean encode_fields
#axiomsClean fields_injective
#axiomsClean finite_projection_iff
#axiomsClean special_decode
#axiomsClean subnormal_or_zero_decode
#axiomsClean decode_boundaries
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.Binary64Words

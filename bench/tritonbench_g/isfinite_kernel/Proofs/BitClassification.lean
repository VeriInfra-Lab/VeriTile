import VeriTile.Triton.Core.Ast
import VeriTile.Triton.Float.BitValue
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

/-! Source-side bit classification for the two libdevice paths used by
the pinned isfinite helper. All bit patterns are included. This file neither
changes the real arithmetic carrier nor supplies a native refinement axiom. -/
namespace VeriTile.Bench.TritonBenchG.IsfiniteKernel.BitClassification
open VeriTile.Triton

inductive Classification where
  | finite | infinity | nan
  deriving DecidableEq, Repr

def exponent32 (x : Float32Bits) : Nat := (x.bits.toNat / 2^23) % 256
def fraction32 (x : Float32Bits) : Nat := x.bits.toNat % 2^23
def exponent64 (x : Float64Bits) : Nat := (x.bits.toNat / 2^52) % 2048
def fraction64 (x : Float64Bits) : Nat := x.bits.toNat % 2^52

def classify (exponent fraction maximum : Nat) : Classification :=
  if exponent = maximum then
    if fraction = 0 then .infinity else .nan
  else .finite

def classify32 (x : Float32Bits) : Classification := classify (exponent32 x) (fraction32 x) 255
def classify64 (x : Float64Bits) : Classification := classify (exponent64 x) (fraction64 x) 2047

def finite32 (x : Float32Bits) : Bool := decide (exponent32 x ≠ 255)
def finite64 (x : Float64Bits) : Bool := decide (exponent64 x ≠ 2047)

/-- The existing format-indexed bf16/fp32 carrier has the same eight-bit
exponent field. Classification never consults its rational magnitude. -/
def bitFinite {f : FP.Format} (x : FP.Value f) : Bool := decide (x.exponent ≠ 255)
def bitClassify {f : FP.Format} (x : FP.Value f) : Classification :=
  classify x.exponent x.fraction 255

def decodedClass : FP.Decoded → Classification
  | .finite .. => .finite
  | .infinity .. => .infinity
  | .nan .. => .nan

theorem classify_finite (e f maximum : Nat) :
    classify e f maximum = .finite ↔ e ≠ maximum := by
  by_cases he : e = maximum <;> by_cases hf : f = 0 <;> simp [classify, he, hf]

theorem classify_infinity (e f maximum : Nat) :
    classify e f maximum = .infinity ↔ e = maximum ∧ f = 0 := by
  by_cases he : e = maximum <;> by_cases hf : f = 0 <;> simp [classify, he, hf]

theorem classify_nan (e f maximum : Nat) :
    classify e f maximum = .nan ↔ e = maximum ∧ f ≠ 0 := by
  by_cases he : e = maximum <;> by_cases hf : f = 0 <;> simp [classify, he, hf]

/-- All payloads, signs and both existing formats are covered by this theorem;
in particular, finite includes both exponent-zero and ordinary encodings. -/
theorem bit_decode_class {f : FP.Format} (x : FP.Value f) :
    decodedClass x.decode = bitClassify x := by
  by_cases he : x.exponent = 255 <;> by_cases hf : x.fraction = 0 <;>
    simp [FP.Value.decode, decodedClass, bitClassify, classify, he, hf]

theorem bitFinite_correct {f : FP.Format} (x : FP.Value f) :
    bitFinite x = true ↔ decodedClass x.decode = .finite := by
  rw [bit_decode_class, bitClassify, classify_finite]
  simp [bitFinite]

theorem bitFinite_eq_projection {f : FP.Format} (x : FP.Value f) :
    bitFinite x = x.toRat?.isSome := by
  by_cases he : x.exponent = 255 <;> by_cases hf : x.fraction = 0 <;>
    simp [bitFinite, FP.Value.toRat?, FP.Value.decode, he, hf]

theorem bitFinite_zero_or_subnormal {f : FP.Format} (x : FP.Value f)
    (h : x.exponent = 0) : bitFinite x = true := by
  simp [bitFinite, h]

theorem bitFinite_normal {f : FP.Format} (x : FP.Value f)
    (h : x.exponent < 255) : bitFinite x = true := by
  simp [bitFinite, Nat.ne_of_lt h]

theorem bitFinite_infinity_or_nan {f : FP.Format} (x : FP.Value f)
    (h : x.exponent = 255) : bitFinite x = false := by
  simp [bitFinite, h]

theorem binary64_zero_or_subnormal (x : Float64Bits)
    (h : exponent64 x = 0) : finite64 x = true := by
  simp [finite64, h]

theorem binary64_normal (x : Float64Bits)
    (h : exponent64 x < 2047) : finite64 x = true := by
  simp [finite64, Nat.ne_of_lt h]

theorem binary64_infinity_or_nan (x : Float64Bits)
    (h : exponent64 x = 2047) : finite64 x = false := by
  simp [finite64, h]

/-- The existing fp32 Value and the source load's Float32Bits agree on their
complete payload classification, without a rational conversion. -/
theorem bitFinite_fp32 (x : FP.Value .fp32) :
    bitFinite x = finite32 ⟨x.bits⟩ := rfl

theorem finite32_correct (x : Float32Bits) : finite32 x = true ↔ classify32 x = .finite := by
  rw [classify32, classify_finite]
  simp [finite32]

theorem finite64_correct (x : Float64Bits) : finite64 x = true ↔ classify64 x = .finite := by
  rw [classify64, classify_finite]
  simp [finite64]

theorem finite32_eq_decode (x : Float32Bits) : finite32 x = x.decodeRat.isSome := by
  by_cases hz : (x.bits.toNat / 2^23) % 256 = 0
  · have hi : (x.bits.toNat / 2^23) % 256 ≠ 255 := by omega
    simp only [finite32, exponent32, Float32Bits.decodeRat, ne_eq, hz, hi,
      ite_true, ite_false, not_true_eq_false, not_false_eq_true,
      decide_true, decide_false, Option.isSome] <;> norm_num
  · by_cases hi : (x.bits.toNat / 2^23) % 256 = 255 <;>
      simp only [finite32, exponent32, Float32Bits.decodeRat, ne_eq, hz, hi,
        ite_true, ite_false, not_true_eq_false, not_false_eq_true,
        decide_true, decide_false, Option.isSome] <;> norm_num

def bits32 (n : Nat) : Float32Bits := ⟨BitVec.ofNat 32 n⟩
def bits64 (n : Nat) : Float64Bits := ⟨BitVec.ofNat 64 n⟩

theorem bfloat16_examples :
    bitFinite (FP.Value.ofBits .bf16 0x0000) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0x8000) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0x0001) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0x8001) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0x3f80) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0xbf80) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0x7f7f) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0xff7f) = true ∧
    bitFinite (FP.Value.ofBits .bf16 0x7f80) = false ∧
    bitFinite (FP.Value.ofBits .bf16 0xff80) = false ∧
    bitFinite (FP.Value.ofBits .bf16 0x7fc1) = false ∧
    bitFinite (FP.Value.ofBits .bf16 0x7f81) = false ∧
    bitFinite (FP.Value.ofBits .bf16 0xffc1) = false ∧
    bitFinite (FP.Value.ofBits .bf16 0xff81) = false := by decide

/-- Signed zeros, minimum subnormals and maximum finite encodings all pass;
positive/negative infinity and quiet/signaling NaNs all fail. -/
theorem binary32_examples :
    finite32 (bits32 0x00000000) = true ∧
    finite32 (bits32 0x80000000) = true ∧
    finite32 (bits32 0x00000001) = true ∧
    finite32 (bits32 0x80000001) = true ∧
    finite32 (bits32 0x3f800000) = true ∧
    finite32 (bits32 0xbf800000) = true ∧
    finite32 (bits32 0x7f7fffff) = true ∧
    finite32 (bits32 0xff7fffff) = true ∧
    finite32 (bits32 0x7f800000) = false ∧
    finite32 (bits32 0xff800000) = false ∧
    finite32 (bits32 0x7fc00001) = false ∧
    finite32 (bits32 0x7f800001) = false ∧
    finite32 (bits32 0xffc00001) = false ∧
    finite32 (bits32 0xff800001) = false := by decide

theorem binary64_examples :
    finite64 (bits64 0x0000000000000000) = true ∧
    finite64 (bits64 0x8000000000000000) = true ∧
    finite64 (bits64 0x0000000000000001) = true ∧
    finite64 (bits64 0x8000000000000001) = true ∧
    finite64 (bits64 0x3ff0000000000000) = true ∧
    finite64 (bits64 0xbff0000000000000) = true ∧
    finite64 (bits64 0x7fefffffffffffff) = true ∧
    finite64 (bits64 0xffefffffffffffff) = true ∧
    finite64 (bits64 0x7ff0000000000000) = false ∧
    finite64 (bits64 0xfff0000000000000) = false ∧
    finite64 (bits64 0x7ff8000000000001) = false ∧
    finite64 (bits64 0x7ff0000000000001) = false ∧
    finite64 (bits64 0xfff8000000000001) = false ∧
    finite64 (bits64 0xfff0000000000001) = false := by decide

/-- The existing binary64 real decoder is partial even on finite values, so
its Option.isSome cannot implement the source's classification. -/
theorem binary64_decoder_is_not_finiteness :
    finite64 (bits64 0) = true ∧ (bits64 0).decodeRat = none ∧
    finite64 (bits64 1) = true ∧ (bits64 1).decodeRat = none := by decide

#axiomsClean classify_finite
#axiomsClean classify_infinity
#axiomsClean classify_nan
#axiomsClean bit_decode_class
#axiomsClean bitFinite_correct
#axiomsClean bitFinite_eq_projection
#axiomsClean bitFinite_zero_or_subnormal
#axiomsClean bitFinite_normal
#axiomsClean bitFinite_infinity_or_nan
#axiomsClean binary64_zero_or_subnormal
#axiomsClean binary64_normal
#axiomsClean binary64_infinity_or_nan
#axiomsClean bitFinite_fp32
#axiomsClean bfloat16_examples
#axiomsClean finite32_correct
#axiomsClean finite64_correct
#axiomsClean finite32_eq_decode
#axiomsClean binary32_examples
#axiomsClean binary64_examples
#axiomsClean binary64_decoder_is_not_finiteness
end VeriTile.Bench.TritonBenchG.IsfiniteKernel.BitClassification

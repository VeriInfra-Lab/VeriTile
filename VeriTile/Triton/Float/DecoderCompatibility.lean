import VeriTile.Triton.Core.Ast
import VeriTile.Triton.Float.BitValue
import Mathlib.Tactic

/-!
Agreement between the algorithm projection of binary32 constants and the
concrete software-value decoder. This module is deliberately above both
layers: Core.Ast does not import the floating-point implementation.
-/
namespace VeriTile.Triton.FP.DecoderCompatibility

theorem pow2_eq_zpow (e : Int) : FP.pow2 e = (2:Rat)^e := by
  cases e with
  | ofNat n =>
      simp only [FP.pow2, Int.ofNat_eq_natCast,
        Int.toNat_natCast, zpow_natCast]
      change (((2:Nat)^n : Nat) : Rat) = _
      norm_cast
  | negSucc n =>
      rw [FP.pow2, if_pos (by omega)]
      change mkRat 1 (2^(n+1)) = (2:Rat)^(-((n+1:Nat):Int))
      rw [zpow_neg, zpow_natCast, Rat.mkRat_eq_divInt, Rat.divInt_eq_div]
      simp only [Int.cast_one, Nat.cast_pow, Nat.cast_ofNat, one_div]
      norm_cast

theorem scale_formula (q : Rat) (e : Int) :
    (if 0 ≤ e then q * (2:Rat)^e.toNat else q / (2:Rat)^(-e).toNat) = q * (2:Rat)^e := by
  cases e with
  | ofNat n => simp
  | negSucc n =>
      rw [if_neg (by omega)]
      change q / (2:Rat)^(n+1) = q * (2:Rat)^(-((n+1:Nat):Int))
      rw [zpow_neg, zpow_natCast, div_eq_mul_inv]

/-- The previous normal-only decoder, retained here as a compatibility oracle. -/
def legacyNormalDecodeRat (x : Float32Bits) : Option Rat :=
  let n := x.bits.toNat
  let sign := n / 2^31
  let exp : Nat := (n / 2^23) % 256
  let frac := n % 2^23
  if exp = 0 ∨ exp = 255 then none else
    let significand : Rat := (2^23 + frac : Nat)
    let signed := if sign = 0 then significand else -significand
    let e := (exp : Int) - 150
    some (if 0 ≤ e then signed * (2 : Rat)^e.toNat
      else signed / (2 : Rat)^(-e).toNat)

/-- All nonzero-exponent results, including NaN/Inf rejection, are unchanged. -/
theorem normal_unchanged (x : Float32Bits)
    (h : (x.bits.toNat / 2^23) % 256 ≠ 0) :
    Float32Bits.decodeRat x = legacyNormalDecodeRat x := by
  simp only [Float32Bits.decodeRat, legacyNormalDecodeRat, h, false_or, ↓reduceIte]
  rfl

theorem core_decode_formula (x : Float32Bits) :
    Float32Bits.decodeRat x =
      let n : Nat := x.bits.toNat
      let e : Nat := (n / 2^23) % 256
      let sig : Rat := (if e = 0 then n % 2^23 else 2^23 + n % 2^23 : Nat)
      if e = 255 then none
      else some ((if n / 2^31 = 0 then sig else -sig) *
        (2:Rat)^(if e = 0 then -149 else (e:Int)-150)) := by
  dsimp only
  by_cases hz : x.bits.toNat / 2^23 % 256 = 0
  · simp only [Float32Bits.decodeRat, hz, ↓reduceIte]
    apply congrArg some
    exact scale_formula _ (-149)
  · by_cases hi : x.bits.toNat / 2^23 % 256 = 255
    · simp only [Float32Bits.decodeRat, hz, ↓reduceIte]
      simp only [hi, ↓reduceIte]
    · simp only [Float32Bits.decodeRat, hz, hi, ↓reduceIte]
      apply congrArg some
      exact scale_formula _ _

theorem value_decode_formula (v : FP.Value .fp32) :
    v.toRat? =
      let sig : Rat := (if v.exponent = 0 then v.fraction else 2^23 + v.fraction : Nat)
      if v.exponent = 255 then none
      else some ((if v.sign then -sig else sig) *
        (2:Rat)^(if v.exponent = 0 then -149 else (v.exponent:Int)-150)) := by
  dsimp only
  by_cases hi : v.exponent = 255
  · have hbi : (v.exponent == 255) = true := by simp only [hi, beq_self_eq_true]
    unfold FP.Value.toRat? FP.Value.decode
    try dsimp only
    erw [hbi, if_pos hi]
    erw [if_pos (rfl : true = true)]
    by_cases hf : (v.fraction == 0) = true
    · erw [if_pos hf]
    · erw [if_neg hf]
  · have hbi : (v.exponent == 255) = false := by exact beq_eq_false_iff_ne.mpr hi
    unfold FP.Value.toRat? FP.Value.decode
    try dsimp only
    erw [hbi, if_neg hi]
    try dsimp only
    by_cases hz : v.exponent = 0
    · have hbz : (v.exponent == 0) = true := by simp only [hz, beq_self_eq_true]
      erw [hbz, if_pos hz]
      try dsimp only
      change some (if v.sign then -(Rat.ofInt (v.fraction: Int) * FP.pow2 (-149))
        else Rat.ofInt (v.fraction:Int) * FP.pow2 (-149)) = _
      rw [pow2_eq_zpow]
      cases v.sign <;> simp only [Bool.false_eq_true, ↓reduceIte]
      · erw [if_pos hz]; rfl
      · erw [if_pos hz]; exact congrArg some (neg_mul _ _).symm
    · have hbz : (v.exponent == 0) = false := by exact beq_eq_false_iff_ne.mpr hz
      erw [hbz, if_neg hz]
      try dsimp only
      have heq : (v.exponent:Int)-127-23 = (v.exponent:Int)-150 := by omega
      change some (if v.sign then -(Rat.ofInt ((2^23 + v.fraction:Nat):Int) *
        FP.pow2 ((v.exponent:Int)-127-23)) else
        Rat.ofInt ((2^23 + v.fraction:Nat):Int) * FP.pow2 ((v.exponent:Int)-127-23)) = _
      rw [pow2_eq_zpow, heq]
      cases v.sign <;> simp only [Bool.false_eq_true, ↓reduceIte]
      · erw [if_neg hz]; rfl
      · erw [if_neg hz]; exact congrArg some (neg_mul _ _).symm

theorem software_decode_formula (x : Float32Bits) :
    (FP.Value.mk x.bits : FP.Value .fp32).toRat? =
      let n : Nat := x.bits.toNat
      let e : Nat := (n / 2^23) % 256
      let sig : Rat := (if e = 0 then n % 2^23 else 2^23 + n % 2^23 : Nat)
      if e = 255 then none
      else some ((if n / 2^31 = 0 then sig else -sig) *
        (2:Rat)^(if e = 0 then -149 else (e:Int)-150)) := by
  rw [value_decode_formula]
  unfold FP.Value.exponent FP.Value.fraction FP.Value.sign FP.Value.raw FP.Format.fractionBits
  dsimp only
  simp only [show 23+8=31 from rfl, bne_iff_ne, ite_not]
  rfl

/-- Universal agreement for every payload, including signed zeros, subnormals,
normals, infinities and every NaN payload. -/
theorem equals_software_decoder (x : Float32Bits) :
    Float32Bits.decodeRat x = (FP.Value.mk x.bits : FP.Value .fp32).toRat? := by
  rw [core_decode_formula, software_decode_formula]

/-- Concrete constants use the same finite rational projection as the software
value layer, without introducing a default for an unsupported payload. -/
theorem constToAlgorithm_of_software (x : Float32Bits) (q : Rat)
    (h : (FP.Value.mk x.bits : FP.Value .fp32).toRat? = some q) :
    ComputeOp.constToAlgorithm? .fp32 x = .ok (Op.const (q : ℝ)) := by
  simp only [ComputeOp.constToAlgorithm?, equals_software_decoder, h]

theorem positive_zero : Float32Bits.decodeRat ⟨0x00000000#32⟩ = some 0 := by decide +kernel
theorem negative_zero : Float32Bits.decodeRat ⟨0x80000000#32⟩ = some 0 := by decide +kernel
theorem smallest_subnormal : Float32Bits.decodeRat ⟨0x00000001#32⟩ = some ((1:Rat)/2^149) := by decide +kernel
theorem negative_subnormal : Float32Bits.decodeRat ⟨0x80000001#32⟩ = some (-((1:Rat)/2^149)) := by decide +kernel
theorem largest_subnormal : Float32Bits.decodeRat ⟨0x007fffff#32⟩ = some ((8388607:Rat)/2^149) := by decide +kernel
theorem smallest_normal : Float32Bits.decodeRat ⟨0x00800000#32⟩ = some ((1:Rat)/2^126) := by decide +kernel
theorem positive_infinity : Float32Bits.decodeRat ⟨0x7f800000#32⟩ = none := by decide +kernel
theorem negative_infinity : Float32Bits.decodeRat ⟨0xff800000#32⟩ = none := by decide +kernel
theorem nan_payload : Float32Bits.decodeRat ⟨0x7fc00001#32⟩ = none := by decide +kernel

theorem const_positive_zero :
    ComputeOp.constToAlgorithm? .fp32 ⟨0x00000000#32⟩ = .ok (Op.const 0) := by
  simp [ComputeOp.constToAlgorithm?, positive_zero]

theorem const_negative_zero :
    ComputeOp.constToAlgorithm? .fp32 ⟨0x80000000#32⟩ = .ok (Op.const 0) := by
  simp [ComputeOp.constToAlgorithm?, negative_zero]

theorem const_smallest_subnormal :
    ComputeOp.constToAlgorithm? .fp32 ⟨0x00000001#32⟩ =
      .ok (Op.const (((1:Rat)/2^149 : Rat) : ℝ)) := by
  simp only [ComputeOp.constToAlgorithm?, smallest_subnormal]

theorem bitcast_negative_zero :
    (ComputeOp.bitcast .uint32 .fp32 rfl
      (.const ⟨0x80000000#32⟩)).toAlgorithm? = .ok (Op.const 0) := by
  simpa only [ComputeOp.toAlgorithm?, ComputeOp.constOpToAlgorithm?,
    ComputeOp.constPayload?, ComputeOp.bitcastPayload] using const_negative_zero

theorem const_nan_rejected :
    ComputeOp.constToAlgorithm? .fp32 ⟨0x7fc00001#32⟩ =
      .error (.unsupportedBitcast "unsupported fp32 decode") := by
  simp only [ComputeOp.constToAlgorithm?, nan_payload]

-- Binary64 retains its existing normal-only domain.
theorem fp64_zero_unchanged : Float64Bits.decodeRat ⟨0x0000000000000000#64⟩ = none := by decide +kernel

theorem fp64_subnormal_unchanged : Float64Bits.decodeRat ⟨0x0000000000000001#64⟩ = none := by decide +kernel

theorem fp64_one_unchanged : Float64Bits.decodeRat ⟨0x3ff0000000000000#64⟩ = some 1 := by decide +kernel

#print axioms normal_unchanged
#print axioms equals_software_decoder
#print axioms constToAlgorithm_of_software
#print axioms bitcast_negative_zero
end VeriTile.Triton.FP.DecoderCompatibility

import VeriTile.Triton.Float.ScalarBackend

/-! Integer-valued literals in the strict FP32 software profile use the same
nearest-even rounder as integer-to-FP32 conversion. Other real literals and
other precision profiles retain the supplied implementation. This fixes a
concrete software interpretation; it is not an axiom about arbitrary ports. -/
namespace VeriTile.Triton.FP.Software
open Structural

noncomputable def integerLiteralFP32 (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (x : ℝ) : FP.Value .fp32 :=
  by
    classical
    exact
      if fp32Context p d then
        if h : ∃ n : Int, (n : ℝ) = x then
          FP.round .fp32 .ieeeValue (Rat.ofInt (Classical.choose h))
        else rest.literal p d x
      else rest.literal p d x

@[simp] theorem integerLiteralFP32_int (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (h : fp32Context p d = true)
    (n : Int) :
    integerLiteralFP32 rest p d (n : ℝ) = FP.round .fp32 .ieeeValue (Rat.ofInt n) := by
  classical
  have existsInt : ∃ m : Int, (m : ℝ) = (n : ℝ) := ⟨n,rfl⟩
  have chosen : Classical.choose existsInt = n := by
    exact_mod_cast Classical.choose_spec existsInt
  simp [integerLiteralFP32,h,existsInt,chosen]

@[simp] theorem integerLiteralFP32_zero (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (h : fp32Context p d = true) :
    integerLiteralFP32 rest p d 0 = FP.Value.zero .fp32 := by
  calc
    integerLiteralFP32 rest p d 0 = FP.round .fp32 .ieeeValue (Rat.ofInt 0) := by
      simpa only [Int.cast_zero] using integerLiteralFP32_int rest p d h 0
    _ = FP.Value.zero .fp32 := rfl

@[simp] theorem integerLiteralFP32_one (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (h : fp32Context p d = true) :
    integerLiteralFP32 rest p d 1 = FP.Value.ofBits .fp32 0x3f800000 := by
  calc
    integerLiteralFP32 rest p d 1 = FP.round .fp32 .ieeeValue (Rat.ofInt 1) := by
      simpa only [Int.cast_one] using integerLiteralFP32_int rest p d h 1
    _ = FP.Value.ofBits .fp32 0x3f800000 := by decide +kernel

theorem integerLiteralFP32_unsupported (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (x : ℝ)
    (h : ¬ ∃ n : Int, (n : ℝ) = x) :
    integerLiteralFP32 rest p d x = rest.literal p d x := by
  simp [integerLiteralFP32,h]

#axiomsClean integerLiteralFP32_int
#axiomsClean integerLiteralFP32_zero
#axiomsClean integerLiteralFP32_one
#axiomsClean integerLiteralFP32_unsupported
end VeriTile.Triton.FP.Software

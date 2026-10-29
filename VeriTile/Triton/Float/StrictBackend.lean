import VeriTile.Triton.Float.IntegerLiteralBackend
import VeriTile.Triton.Float.MaximumExact
import VeriTile.Triton.Float.ExponentialExact

/-! The fixed software profile used to certify shared strict FP32 rules.
Arithmetic, integer conversion and integer-valued literals use the same
nearest-even rounder without flushing. Maximum
uses the existing maxnum profile. The two exponential primitives use their
separate documented special-value dispatches, with arbitrary finite residual
algorithms and an explicit canonical NaN choice. In this adapter the `rest`
`.exp` slot supplies the residual ex2 algorithm, whose input has already been
multiplied by log2(e); it is not a second natural exponential. Other operations remain in
`rest`; this definition supplies neither GPU conformance nor empirical laws. -/
namespace VeriTile.Triton.FP.Software
open Structural

instance strictValueInhabited : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

def strictTLExpResidual (rest : Algebra (FP.Value .fp32)) :
    ExponentialExact.Residual .ptxEx2Approx :=
  ⟨rest.unary (some .fp32) .exp,
    fun _ => ⟨FP.Value.qNaN .fp32, by decide +kernel⟩⟩

def strictLibdeviceExpResidual (rest : Algebra (FP.Value .fp32)) :
    ExponentialExact.Residual .libdeviceExpf :=
  ⟨rest.unary (some .fp32) .libdeviceExp,
    fun _ => ⟨FP.Value.qNaN .fp32, by decide +kernel⟩⟩

noncomputable def strictFP32 (rest : Algebra (FP.Value .fp32)) : Algebra (FP.Value .fp32) :=
  let scalar := scalarFP32 rest .ieeeValue
  { scalar with
    literal := integerLiteralFP32 scalar
    binary := fun p d op a b =>
      if fp32Context p d then
        match op with
        | .max => MaximumExact.maximumValue a b
        | _ => scalar.binary p d op a b
      else scalar.binary p d op a b
    fromNat := fun p n =>
      if p = some .fp32 then FP.round .fp32 .ieeeValue (n : Rat) else rest.fromNat p n
    fromInt := fun p n =>
      if p = some .fp32 then FP.round .fp32 .ieeeValue (Rat.ofInt n) else rest.fromInt p n
    unary := fun p op x =>
      if p = some .fp32 then
        match op with
        | .exp => ExponentialExact.tlExp (strictTLExpResidual rest) x
        | .libdeviceExp => ExponentialExact.libdeviceExp (strictLibdeviceExpResidual rest) x
        | _ => scalar.unary p op x
      else scalar.unary p op x }

@[simp] theorem strictFP32_add (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (a b : FP.Value .fp32)
    (h : fp32Context p d = true) :
    (strictFP32 rest).binary p d .add a b = FP.add .ieeeValue a b := by
  simp [strictFP32, scalarFP32, h]

@[simp] theorem strictFP32_mul (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (a b : FP.Value .fp32)
    (h : fp32Context p d = true) :
    (strictFP32 rest).binary p d .mul a b = FP.mul .ieeeValue a b := by
  simp [strictFP32, scalarFP32, h]

@[simp] theorem strictFP32_sub (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (a b : FP.Value .fp32)
    (h : fp32Context p d = true) :
    (strictFP32 rest).binary p d .sub a b = FP.sub .ieeeValue a b := by
  simp [strictFP32, scalarFP32, h]

@[simp] theorem strictFP32_exp (rest : Algebra (FP.Value .fp32)) (x : FP.Value .fp32) :
    (strictFP32 rest).unary (some .fp32) .exp x =
      ExponentialExact.tlExp (strictTLExpResidual rest) x := by
  simp [strictFP32]

@[simp] theorem strictFP32_libdeviceExp (rest : Algebra (FP.Value .fp32)) (x : FP.Value .fp32) :
    (strictFP32 rest).unary (some .fp32) .libdeviceExp x =
      ExponentialExact.libdeviceExp (strictLibdeviceExpResidual rest) x := by
  simp [strictFP32]

@[simp] theorem strictFP32_fromNat (rest : Algebra (FP.Value .fp32)) (n : Nat) :
    (strictFP32 rest).fromNat (some .fp32) n = FP.round .fp32 .ieeeValue (n : Rat) := by
  simp [strictFP32]

@[simp] theorem strictFP32_max (rest : Algebra (FP.Value .fp32))
    (p : Option ComputeDType) (d : FloatDType) (a b : FP.Value .fp32)
    (h : fp32Context p d = true) :
    (strictFP32 rest).binary p d .max a b = MaximumExact.maximumValue a b := by
  simp [strictFP32, h]

end VeriTile.Triton.FP.Software

import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-!
A concrete software interpretation of the fp32 scalar fragment. Each add,
subtract, multiply and divide rounds at its own result using `ScalarOps`.
Same-format casts and fp32 loads preserve bits, including NaN payloads and
signed zero. This is the unfused operation graph, not GPU conformance.

The supplied `rest` keeps every other operation explicit. In particular this
adapter supplies no transcendental, reduction, dot, mixed-format conversion,
or arbitrary-real-literal implementation. Kernel contracts instantiated with
it must show that their result depends only on the covered fragment. No
cast-idempotence or arithmetic rewrite law is assumed.

An explicit `some .fp32` compute operation may have the `.real` carrier tag
because `ComputeDType.fp32.eraseDType = .real`; that explicit precision is
supported too. The untagged `(none, .real)` profile remains in `rest`.
-/
namespace VeriTile.Triton.FP.Software

def fp32Context : Option ComputeDType → FloatDType → Bool
  | none, .fp32 | some .fp32, .fp32 | some .fp32, .real => true
  | _, _ => false

noncomputable def scalarFP32
    (rest : Structural.Algebra (FP.Value .fp32)) (cfg : Config := .ieeeValue) :
    Structural.Algebra (FP.Value .fp32) :=
  { rest with
    literal := fun p d x =>
      if fp32Context p d then
        if x = 0 then FP.Value.zero .fp32
        else if x = 1 then FP.Value.ofBits .fp32 0x3f800000
        else rest.literal p d x
      else rest.literal p d x
    negInf := FP.Value.infinity .fp32 true
    binary := fun p d op a b =>
      if fp32Context p d then
        match op with
        | .add => FP.add cfg a b
        | .sub => FP.sub cfg a b
        | .mul => FP.mul cfg a b
        | .div => FP.div cfg a b
        | _ => rest.binary p d op a b
      else rest.binary p d op a b
    unary := fun p op x =>
      match op with
      | .neg dtype => if fp32Context p dtype then x.neg else rest.unary p op x
      | .abs dtype => if fp32Context p dtype then x.abs else rest.unary p op x
      | _ => rest.unary p op x
    cast := fun p src dst x =>
      if (src = .fp32 ∧ dst = .fp32) ∨
          (p = some .fp32 ∧ src = .real ∧ dst = .real) then x
      else rest.cast p src dst x
    fp32Bits := fun bits => ⟨bits.bits⟩
    fp32Load := id }

@[simp] theorem scalarFP32_zero (rest : Structural.Algebra (FP.Value .fp32))
    (cfg : Config) (p : Option ComputeDType) (h : fp32Context p .fp32 = true) :
    (scalarFP32 rest cfg).literal p .fp32 0 = FP.Value.zero .fp32 := by
  simp [scalarFP32, h]

@[simp] theorem scalarFP32_one (rest : Structural.Algebra (FP.Value .fp32))
    (cfg : Config) (p : Option ComputeDType) (h : fp32Context p .fp32 = true) :
    (scalarFP32 rest cfg).literal p .fp32 1 = FP.Value.ofBits .fp32 0x3f800000 := by
  simp [scalarFP32, h]

@[simp] theorem scalarFP32_add (rest : Structural.Algebra (FP.Value .fp32))
    (cfg : Config) (p : Option ComputeDType) (a b : FP.Value .fp32)
    (h : fp32Context p .fp32 = true) :
    (scalarFP32 rest cfg).binary p .fp32 .add a b = FP.add cfg a b := by
  simp [scalarFP32, h]

@[simp] theorem scalarFP32_mul (rest : Structural.Algebra (FP.Value .fp32))
    (cfg : Config) (p : Option ComputeDType) (a b : FP.Value .fp32)
    (h : fp32Context p .fp32 = true) :
    (scalarFP32 rest cfg).binary p .fp32 .mul a b = FP.mul cfg a b := by
  simp [scalarFP32, h]

@[simp] theorem scalarFP32_neg (rest : Structural.Algebra (FP.Value .fp32))
    (cfg : Config) (p : Option ComputeDType) (dtype : FloatDType) (a : FP.Value .fp32)
    (h : fp32Context p dtype = true) :
    (scalarFP32 rest cfg).unary p (.neg dtype) a = a.neg := by
  simp [scalarFP32, h]

@[simp] theorem scalarFP32_abs (rest : Structural.Algebra (FP.Value .fp32))
    (cfg : Config) (p : Option ComputeDType) (dtype : FloatDType) (a : FP.Value .fp32)
    (h : fp32Context p dtype = true) :
    (scalarFP32 rest cfg).unary p (.abs dtype) a = a.abs := by
  simp [scalarFP32, h]

@[simp] theorem scalarFP32_cast_same (rest : Structural.Algebra (FP.Value .fp32))
    (cfg : Config) (p : Option ComputeDType) (a : FP.Value .fp32) :
    (scalarFP32 rest cfg).cast p .fp32 .fp32 a = a := by
  simp [scalarFP32]

#axiomsClean scalarFP32_zero
#axiomsClean scalarFP32_one
#axiomsClean scalarFP32_add
#axiomsClean scalarFP32_mul
#axiomsClean scalarFP32_neg
#axiomsClean scalarFP32_abs
#axiomsClean scalarFP32_cast_same
end VeriTile.Triton.FP.Software

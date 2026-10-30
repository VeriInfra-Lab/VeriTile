import VeriTile.Torch.Tensor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous
noncomputable def formula (xs : Nat → ℝ) (i : Nat) : ℝ := Real.sin (xs i)

abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

/-- Exact alias is safe. Shifted overlap of active byte spans can race. -/
def Compatible (x out : TensorMetadata) : Prop :=
  x.dataPtr = out.dataPtr ∨
  x.dataPtr.toNat + 4*x.numel ≤ out.dataPtr.toNat ∨
  out.dataPtr.toNat + 4*x.numel ≤ x.dataPtr.toNat
instance (x out : TensorMetadata) : Decidable (Compatible x out) := inferInstanceAs (Decidable (_ ∨ _ ∨ _))

structure InputContract (x out : TensorMetadata) : Prop where
  layout : x.layout = .strided ∧ out.layout = .strided
  dtype : x.dtype = .floating .fp32 ∧ out.dtype = .floating .fp32
  cuda : x.device.kind = "cuda" ∧ out.device.kind = "cuda"
  device : x.device = out.device
  capacity : x.numel ≤ out.numel
  contiguous : x.isContiguous = Bool.true ∧ out.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false ∧ out.isNeg = Bool.false
  extent : x.numel < 2^31
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ out.dataPtr.toNat % 4 = 0
  overlap : 0 < x.numel → Compatible x out

inductive InputError where
  | tensors | layout | dtype | cuda | device | capacity | contiguous | negative | extent | alignment | overlap
  deriving DecidableEq, Repr

def checkTensorInputs (x out : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || out.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || out.dtype != .floating .fp32 then .error .dtype
  else if !(x.isCuda && out.isCuda) then .error .cuda
  else if x.device != out.device then .error .device
  else if out.numel < x.numel then .error .capacity
  else if !(x.isContiguous && out.isContiguous) then .error .contiguous
  else if x.isNeg || out.isNeg then .error .negative
  else if x.numel >= 2^31 then .error .extent
  else if x.numel > 0 && (x.dataPtr.toNat % 4 != 0 || out.dataPtr.toNat % 4 != 0) then .error .alignment
  else if x.numel > 0 && !decide (Compatible x out) then .error .overlap
  else .ok ()

def checkInputs : ArgumentMetadata → ArgumentMetadata → Except InputError Unit
  | .tensor x, .tensor out => checkTensorInputs x out
  | _, _ => .error .tensors
def AdmittedArguments : ArgumentMetadata → ArgumentMetadata → Prop
  | .tensor x, .tensor out => InputContract x out
  | _, _ => False
end VeriTile.Bench.Repairs.SinComputation.Contiguous

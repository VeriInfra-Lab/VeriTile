import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

/-! Supported input domain and ordered Python admission. Program definitions
and correctness proofs depend on this contract without imposing a check order
on its named fields. Non-Tensor arguments have no Tensor attributes. -/
namespace VeriTile.Bench.Repairs.DropoutTriton

abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

structure TensorContract (x y : TensorMetadata) : Prop where
  layout : x.layout = .strided ∧ y.layout = .strided
  dtype : x.dtype = .floating .fp32 ∧ y.dtype = .other "int32"
  cuda : x.device.kind = "cuda" ∧ y.device.kind = "cuda"
  sameDevice : x.device = y.device
  sameShape : x.shape = y.shape
  contiguous : x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false ∧ y.isNeg = Bool.false
  extent : x.numel < 2^31
  /-- Scalar CUDA `.b32` accesses require 4-byte natural alignment under
  [PTX ISA 8.7, section 6.4.1](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#addresses-as-operands).
  The add_example load/store audit records scalar `.b32` lowering; empty inputs launch nothing. -/
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ y.dataPtr.toNat % 4 = 0

inductive InputError where
  | tensors | layout | dtype | cuda | device | shape | contiguous | negative | extent | alignment | scalarType | probability
  deriving DecidableEq, Repr

/-- The nine attribute checks after Python's initial isinstance rejection. -/
def checkTensorInputs (x y : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || y.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || y.dtype != .other "int32" then .error .dtype
  else if !(x.isCuda && y.isCuda) then .error .cuda
  else if x.device != y.device then .error .device
  else if x.shape != y.shape then .error .shape
  else if !(x.isContiguous && y.isContiguous) then .error .contiguous
  else if x.isNeg || y.isNeg then .error .negative
  else if x.numel >= 2^31 then .error .extent
  else if x.numel > 0 && (x.dataPtr.toNat % 4 != 0 || y.dataPtr.toNat % 4 != 0) then
    .error .alignment
  else .ok ()

/-- Builtin Python floats are decoded here; `none` denotes NaN or infinity.
Other Python types are rejected, so integer-specialization paths are absent. -/
inductive ScalarArgument where
  | other
  | float (value : Option ℝ)

noncomputable def checkScalar (p : ScalarArgument) : Except InputError Unit := by
  classical
  exact match p with
  | .other => .error .scalarType
  | .float none => .error .probability
  | .float (some q) => if 0 ≤ q ∧ q ≤ 1 then .ok () else .error .probability

def ScalarContract (p : ScalarArgument) : Prop :=
  ∃ q : ℝ, p = .float (some q) ∧ 0 ≤ q ∧ q ≤ 1

structure InputContract (x y : TensorMetadata) (p : ScalarArgument) : Prop where
  tensors : TensorContract x y
  scalar : ScalarContract p

noncomputable def checkInputs (x y : ArgumentMetadata) (p : ScalarArgument) : Except InputError Unit :=
  match x, y with
  | .tensor tx, .tensor ty => do
      checkTensorInputs tx ty
      checkScalar p
  | _, _ => .error .tensors

def AdmittedArguments (x y : ArgumentMetadata) (p : ScalarArgument) : Prop :=
  match x, y with
  | .tensor tx, .tensor ty => InputContract tx ty p
  | _, _ => False

/-- Total Real projection only. A kept p=1 coordinate has no conventional
inverted-dropout value; Correct makes that mathematical domain explicit. -/
noncomputable def expected (x : Nat → ℝ) (keep : Nat → Int) (p : ℝ) (i : Nat) : ℝ :=
  if keep i ≠ 0 then x i / (1-p) else 0

namespace Correct

/-- Ordinary dropout is undefined on a kept p=1 coordinate. Dropped
coordinates are zero even at p=1; the caller supplies the keep mask. -/
noncomputable def mathematicalValue (x : ℝ) (keep : Int) (p : ℝ) : Option ℝ :=
  if keep = 0 then some 0 else if p = 1 then none else some (x / (1-p))

def MathematicalDomain (n : Nat) (keep : Nat → Int) (p : ℝ) : Prop :=
  ∀ i, i < n → keep i ≠ 0 → p ≠ 1

end Correct

end VeriTile.Bench.Repairs.DropoutTriton

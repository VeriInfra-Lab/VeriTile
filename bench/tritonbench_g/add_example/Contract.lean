import VeriTile.Torch.Tensor
import Mathlib.Data.Real.Basic

/-! Supported input domain and ordered Python admission. Program definitions
and correctness proofs depend on this contract without imposing a check order
on its named fields. Non-Tensor arguments have no Tensor attributes. -/
namespace VeriTile.Bench.Repairs.AddExample.Contiguous

abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

structure InputContract (x y : TensorMetadata) : Prop where
  layout : x.layout = .strided ∧ y.layout = .strided
  dtype : x.dtype = .floating .fp32 ∧ y.dtype = .floating .fp32
  cuda : x.device.kind = "cuda" ∧ y.device.kind = "cuda"
  sameDevice : x.device = y.device
  sameShape : x.shape = y.shape
  contiguous : x.isContiguous = Bool.true ∧ y.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false ∧ y.isNeg = Bool.false
  extent : x.numel < 2^31
  /-- Scalar CUDA `.b32` accesses require 4-byte natural alignment under
  [PTX ISA 8.7, section 6.4.1](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#addresses-as-operands).
  `LOAD_STORE_AUDIT.md` records the actual lowering; empty inputs launch nothing. -/
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0 ∧ y.dataPtr.toNat % 4 = 0

inductive InputError where
  | tensors | layout | dtype | cuda | device | shape | contiguous | negative | extent | alignment
  deriving DecidableEq, Repr

/-- The nine attribute checks after Python's initial isinstance rejection. -/
def checkTensorInputs (x y : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || y.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || y.dtype != .floating .fp32 then .error .dtype
  else if !(x.isCuda && y.isCuda) then .error .cuda
  else if x.device != y.device then .error .device
  else if x.shape != y.shape then .error .shape
  else if !(x.isContiguous && y.isContiguous) then .error .contiguous
  else if x.isNeg || y.isNeg then .error .negative
  else if x.numel >= 2^31 then .error .extent
  else if x.numel > 0 && (x.dataPtr.toNat % 4 != 0 || y.dataPtr.toNat % 4 != 0) then
    .error .alignment
  else .ok ()

/-- Classify arguments before reading any Tensor attribute. -/
def checkInputs (x y : ArgumentMetadata) : Except InputError Unit :=
  match x, y with
  | .tensor tx, .tensor ty => checkTensorInputs tx ty
  | _, _ => .error .tensors

/-- Lift the typed input contract to the actual dynamic argument domain. -/
def AdmittedArguments (x y : ArgumentMetadata) : Prop :=
  match x, y with
  | .tensor tx, .tensor ty => InputContract tx ty
  | _, _ => False

/-- Independent mathematical formula. -/
def formula (xs ys : Nat → ℝ) (i : Nat) : ℝ := xs i + ys i

end VeriTile.Bench.Repairs.AddExample.Contiguous

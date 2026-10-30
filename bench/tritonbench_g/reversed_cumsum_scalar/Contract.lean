import VeriTile.Torch.Tensor
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Real.Basic

/-! The checked public input domain. Kernel memory and allocator bindings are
ordinary API laws, separately stated in the proof layer. -/
namespace VeriTile.Bench.Repairs.ReversedCumsumScalar
open VeriTile.Torch
open scoped BigOperators

/-- Independent ordinary suffix sum over the input before execution. -/
noncomputable def mathematical (input : Nat → ℝ) (T t : Nat) : ℝ :=
  ∑ k ∈ (Finset.range T).filter (fun k => t ≤ k), input k

def rows (s : TensorMetadata) : Nat := s.shape.getD 0 0 * s.shape.getD 1 0

structure InputContract (s : TensorMetadata) (dtype : Option DType) : Prop where
  layout : s.layout = .strided
  inputDType : s.dtype = .floating .fp32
  outputDType : dtype = none ∨ dtype = some (.floating .fp32)
  rank : s.shape.length = 3
  cuda : s.device.kind = "cuda"
  contiguous : s.isContiguous = true
  noNeg : s.isNeg = false
  grid : 0 < s.numel → rows s < 2^31
  alignment : 0 < s.numel → s.dataPtr.toNat % 4 = 0

inductive InputError where
  | tensor | layout | inputDType | outputDType | rank | cuda | contiguous | negative | grid | alignment
  deriving DecidableEq, Repr

def checkTensorInput (s : TensorMetadata) (dtype : Option DType) : Except InputError Unit :=
  if s.layout != .strided then .error .layout
  else if s.dtype != .floating .fp32 then .error .inputDType
  else if dtype != none && dtype != some (.floating .fp32) then .error .outputDType
  else if s.shape.length != 3 then .error .rank
  else if !s.isCuda then .error .cuda
  else if !s.isContiguous then .error .contiguous
  else if s.isNeg then .error .negative
  else if s.numel > 0 && rows s >= 2^31 then .error .grid
  else if s.numel > 0 && s.dataPtr.toNat % 4 != 0 then .error .alignment
  else .ok ()

def checkInput (s : ArgumentMetadata) (dtype : Option DType) : Except InputError Unit :=
  match s with
  | .nonTensor => .error .tensor
  | .tensor s => checkTensorInput s dtype

def AdmittedArgument (s : ArgumentMetadata) (dtype : Option DType) : Prop :=
  match s with
  | .nonTensor => False
  | .tensor s => InputContract s dtype

end VeriTile.Bench.Repairs.ReversedCumsumScalar

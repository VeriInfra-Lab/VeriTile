import VeriTile.Torch.Tensor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace VeriTile.Bench.Repairs.CosineCompute
noncomputable def mathematical (n : Nat) (input output : Nat → ℝ) : Prop :=
  ∀ i, i < n → output i = Real.cos (input i)

abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

structure InputContract (x : TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  contiguous : x.isContiguous = true
  noNeg : x.isNeg = false
  extent : x.numel < 2^31
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0

inductive InputError where
  | tensor | layout | dtype | cuda | contiguous | negative | extent | alignment
  deriving DecidableEq, Repr

def checkTensorInput (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if x.numel >= 2^31 then .error .extent
  else if x.numel > 0 && x.dataPtr.toNat % 4 != 0 then .error .alignment
  else .ok ()

def checkInput : ArgumentMetadata → Except InputError Unit
  | .nonTensor => .error .tensor
  | .tensor x => checkTensorInput x

def AdmittedArgument : ArgumentMetadata → Prop
  | .nonTensor => False
  | .tensor x => InputContract x

end VeriTile.Bench.Repairs.CosineCompute

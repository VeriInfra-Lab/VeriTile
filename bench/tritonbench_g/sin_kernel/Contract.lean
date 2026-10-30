import VeriTile.Torch.Tensor
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

namespace VeriTile.Bench.Repairs.SinKernel
noncomputable def formula (xs : Nat → ℝ) (i : Nat) : ℝ := Real.sin (xs i)

abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
structure InputContract (x : TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .floating .fp32
  cuda : x.device.kind = "cuda"
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false
  extent : x.numel < 2^31
  alignment : 0 < x.numel → x.dataPtr.toNat % 4 = 0
inductive InputError where
  | tensor | layout | dtype | cuda | contiguous | negative | extent | alignment
  deriving DecidableEq, Repr
def checkTensorInputs (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if x.numel >= 2^31 then .error .extent
  else if x.numel > 0 && x.dataPtr.toNat % 4 != 0 then .error .alignment
  else .ok ()
def checkInputs : ArgumentMetadata → Except InputError Unit
  | .tensor x => checkTensorInputs x
  | _ => .error .tensor
def AdmittedArguments : ArgumentMetadata → Prop
  | .tensor x => InputContract x
  | _ => False
end VeriTile.Bench.Repairs.SinKernel

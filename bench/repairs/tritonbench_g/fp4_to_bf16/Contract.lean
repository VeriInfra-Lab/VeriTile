import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.Fp4ToBf16.Checked
open VeriTile.Torch

/-- Contiguous packed uint8 Tensor values. The negative-view guard makes
direct storage bytes equal logical packed bytes. No count cutoff is added. -/
structure InputContract (x : TensorMetadata) : Prop where
  layout : x.layout = .strided
  dtype : x.dtype = .other "uint8"
  rank : x.shape ≠ []
  cuda : x.device.kind = "cuda"
  contiguous : x.isContiguous = Bool.true
  noNeg : x.isNeg = Bool.false

inductive InputError where
  | tensor | layout | dtype | rank | cuda | contiguous | negative
  deriving DecidableEq, Repr

def checkTensor (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .other "uint8" then .error .dtype
  else if x.shape = [] then .error .rank
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else .ok ()

def checkInputs : ArgumentMetadata → Except InputError Unit
  | .nonTensor => .error .tensor
  | .tensor x => checkTensor x

def AdmittedArguments : ArgumentMetadata → Prop
  | .nonTensor => False
  | .tensor x => InputContract x
end VeriTile.Bench.Repairs.Fp4ToBf16.Checked

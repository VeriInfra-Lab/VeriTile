import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.AddValue.Contiguous
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
  | .tensor x => checkTensorInput x
  | .nonTensor => .error .tensor

def AdmittedArgument : ArgumentMetadata → Prop
  | .tensor x => InputContract x
  | .nonTensor => False

end VeriTile.Bench.Repairs.AddValue.Contiguous

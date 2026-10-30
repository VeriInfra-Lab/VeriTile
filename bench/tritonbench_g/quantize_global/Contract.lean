import VeriTile.Torch.Tensor
namespace QuantizeGlobalRepair.Direct
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
structure InputContract (x : TensorMetadata) (allFinite : Bool) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
  contiguous : x.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false
  extent : x.numel<2^63
  alignment : 0<x.numel → x.dataPtr.toNat%4=0
  finite : x.numel=0 ∨ allFinite=Bool.true
inductive InputError where
  | tensor | dtype | cuda | contiguous | extent | alignment | finite
  deriving DecidableEq,Repr

def checkTensorInputs (x : TensorMetadata) (allFinite : Bool) : Except InputError Unit :=
  if x.layout != .strided || x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if !x.isContiguous || x.isNeg then .error .contiguous
  else if x.numel≥2^63 then .error .extent
  else if x.numel>0 && x.dataPtr.toNat%4 != 0 then .error .alignment
  else if x.numel != 0 && !allFinite then .error .finite
  else .ok ()
def checkInputs (x : ArgumentMetadata) (allFinite : Bool) : Except InputError Unit :=
  match x with
  | .nonTensor => .error .tensor
  | .tensor t => checkTensorInputs t allFinite
def AdmittedArguments (x : ArgumentMetadata) (allFinite : Bool) : Prop :=
  match x with
  | .nonTensor => False
  | .tensor t => InputContract t allFinite
end QuantizeGlobalRepair.Direct

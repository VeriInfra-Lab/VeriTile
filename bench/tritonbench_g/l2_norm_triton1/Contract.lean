import VeriTile.Torch.Tensor
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.L2NormTriton1
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

/-- The 16-byte pointer guard preserves the reviewed JIT vector/reduction
profiles. It is checked on the actual data pointer, including storage offset. -/
structure InputContract (x : TensorMetadata) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
  rank : x.shape≠[]
  width : 0<x.shape.getLastD 0 ∧ x.shape.getLastD 0≤16384
  contiguous : x.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false
  alignment : 0<x.numel → x.dataPtr.toNat%16=0

inductive InputError where
  | tensor | layout | dtype | cuda | rank | width | contiguous | negative | alignment
  deriving DecidableEq, Repr

def checkTensorInputs (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if x.shape=[] then .error .rank
  else if x.shape.getLastD 0=0 || x.shape.getLastD 0>16384 then .error .width
  else if !x.isContiguous then .error .contiguous
  else if x.isNeg then .error .negative
  else if x.numel>0 && x.dataPtr.toNat%16 != 0 then .error .alignment
  else .ok ()

def checkInputs : ArgumentMetadata → Except InputError Unit
  | .nonTensor => .error .tensor
  | .tensor x => checkTensorInputs x

def AdmittedArguments : ArgumentMetadata → Prop
  | .nonTensor => False
  | .tensor x => InputContract x

end VeriTile.Bench.Repairs.L2NormTriton1

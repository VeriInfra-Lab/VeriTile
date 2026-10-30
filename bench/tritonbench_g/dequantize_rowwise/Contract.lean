import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.DequantizeRowwise
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
def rows (x : TensorMetadata) : Nat := x.shape[0]?.getD 0
def columns (x : TensorMetadata) : Nat := x.shape[1]?.getD 0
structure InputContract (x s : TensorMetadata) : Prop where
  layout : x.layout=.strided ∧ s.layout=.strided
  dtype : x.dtype=.other "int8" ∧ s.dtype=.floating .fp32
  cuda : x.device.kind="cuda" ∧ s.device.kind="cuda"
  sameDevice : x.device=s.device
  rank : x.shape.length=2 ∧ s.shape.length=1
  contiguous : x.isContiguous=Bool.true ∧ s.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false ∧ s.isNeg=Bool.false
  width : columns x≤2^20
  grid : rows x<2^31
  scales : 0<x.numel → rows x≤ s.numel
  alignment : 0<x.numel → s.dataPtr.toNat%4=0
inductive InputError where
  | tensors | layout | dtype | cuda | device | rank | contiguous | width | grid | scales | alignment
  deriving DecidableEq,Repr
def checkTensorInputs (x s : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || s.layout != .strided then .error .layout
  else if x.dtype != .other "int8" || s.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda || !s.isCuda then .error .cuda
  else if x.device != s.device then .error .device
  else if x.shape.length != 2 || s.shape.length != 1 then .error .rank
  else if !x.isContiguous || !s.isContiguous || x.isNeg || s.isNeg then .error .contiguous
  else if columns x>2^20 then .error .width
  else if rows x≥2^31 then .error .grid
  else if x.numel>0 && rows x>s.numel then .error .scales
  else if x.numel>0 && s.dataPtr.toNat%4 != 0 then .error .alignment
  else .ok ()
def checkInputs (x s : ArgumentMetadata) : Except InputError Unit :=
  match x,s with
  | .tensor a,.tensor b => checkTensorInputs a b
  | _,_ => .error .tensors
def AdmittedArguments (x s : ArgumentMetadata) : Prop :=
  match x,s with
  | .tensor a,.tensor b => InputContract a b
  | _,_ => False
end VeriTile.Bench.Repairs.DequantizeRowwise

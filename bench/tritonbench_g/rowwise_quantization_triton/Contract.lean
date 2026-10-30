import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.RowwiseQuantizationTriton
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata
def rows (x : TensorMetadata) : Nat := x.shape[0]?.getD 0
def columns (x : TensorMetadata) : Nat := x.shape[1]?.getD 0
structure InputContract (x : TensorMetadata) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
  rank : x.shape.length=2
  contiguous : x.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false
  width : 0<columns x ∧ columns x≤2^20
  grid : rows x<2^31
  alignment : 0<x.numel → x.dataPtr.toNat%4=0
inductive InputError where
  | tensor | dtype | cuda | rank | contiguous | width | grid | alignment
  deriving DecidableEq,Repr
def checkTensorInputs (x : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || x.dtype != .floating .fp32 then .error .dtype
  else if !x.isCuda then .error .cuda
  else if x.shape.length != 2 then .error .rank
  else if !x.isContiguous || x.isNeg then .error .contiguous
  else if columns x=0 || columns x>2^20 then .error .width
  else if rows x≥2^31 then .error .grid
  else if x.numel>0 && x.dataPtr.toNat%4 != 0 then .error .alignment
  else .ok ()
def checkInputs : ArgumentMetadata → Except InputError Unit
  | .nonTensor => .error .tensor
  | .tensor x => checkTensorInputs x
def AdmittedArguments : ArgumentMetadata → Prop
  | .nonTensor => False
  | .tensor x => InputContract x
end VeriTile.Bench.Repairs.RowwiseQuantizationTriton

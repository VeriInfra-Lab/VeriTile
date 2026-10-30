import bench.repairs.tritonbench_g.l2_norm_triton1.Contract

namespace VeriTile.Bench.Repairs.L2NormTriton2
abbrev TensorMetadata := L2NormTriton1.TensorMetadata
abbrev ArgumentMetadata := L2NormTriton1.ArgumentMetadata
abbrev ForwardContract := L2NormTriton1.InputContract
abbrev checkForward := L2NormTriton1.checkInputs

/-- Backward preserves the original DY-last-axis grouping; X may have another
shape, including scalar shape when DY has one element. -/
structure InputContract (x dy : TensorMetadata) : Prop where
  layout : x.layout=.strided ∧ dy.layout=.strided
  dtype : x.dtype=.floating .fp32 ∧ dy.dtype=.floating .fp32
  cuda : x.device.kind="cuda" ∧ dy.device.kind="cuda"
  sameDevice : x.device=dy.device
  rank : dy.shape≠[]
  width : 0<dy.shape.getLastD 0 ∧ dy.shape.getLastD 0≤16384
  sameCount : x.numel=dy.numel
  contiguous : x.isContiguous=Bool.true ∧ dy.isContiguous=Bool.true
  noNeg : x.isNeg=Bool.false ∧ dy.isNeg=Bool.false
  alignment : 0<x.numel → x.dataPtr.toNat%16=0 ∧ dy.dataPtr.toNat%16=0

inductive InputError where
  | tensors | layout | dtype | cuda | device | rank | width | count | contiguous | negative | alignment
  deriving DecidableEq, Repr

def checkTensorInputs (x dy : TensorMetadata) : Except InputError Unit :=
  if x.layout != .strided || dy.layout != .strided then .error .layout
  else if x.dtype != .floating .fp32 || dy.dtype != .floating .fp32 then .error .dtype
  else if !(x.isCuda && dy.isCuda) then .error .cuda
  else if x.device != dy.device then .error .device
  else if dy.shape=[] then .error .rank
  else if dy.shape.getLastD 0=0 || dy.shape.getLastD 0>16384 then .error .width
  else if x.numel != dy.numel then .error .count
  else if !(x.isContiguous && dy.isContiguous) then .error .contiguous
  else if x.isNeg || dy.isNeg then .error .negative
  else if x.numel>0 && (x.dataPtr.toNat%16 != 0 || dy.dataPtr.toNat%16 != 0) then .error .alignment
  else .ok ()

def checkInputs (x dy : ArgumentMetadata) : Except InputError Unit :=
  match x,dy with
  | .tensor x,.tensor dy => checkTensorInputs x dy
  | _,_ => .error .tensors

def AdmittedArguments (x dy : ArgumentMetadata) : Prop :=
  match x,dy with
  | .tensor x,.tensor dy => InputContract x dy
  | _,_ => False
end VeriTile.Bench.Repairs.L2NormTriton2

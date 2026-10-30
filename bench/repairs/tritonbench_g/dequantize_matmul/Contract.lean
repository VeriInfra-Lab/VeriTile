import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.DequantizeMatmul.Direct
open VeriTile.Torch
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata

/-- Storage identity is the ordinary storage-alias query used by `_overlaps`.
Same-dtype inputs may alias; mixed INT8/FP32 raw storage is rejected. -/
structure TensorInfo where
  metadata : TensorMetadata
  storage : Nat

def rows (x : TensorInfo) := x.metadata.shape[0]?.getD 0
def columns (x : TensorInfo) := x.metadata.shape[1]?.getD 0

def Aligned (x : TensorInfo) : Prop := 0 < x.metadata.numel → x.metadata.dataPtr.toNat%4=0
instance (x : TensorInfo) : Decidable (Aligned x) := inferInstanceAs (Decidable (_ → _))

structure OutputContract (a b scale out : TensorInfo) : Prop where
  layout : out.metadata.layout=.strided
  shape : out.metadata.shape=[rows a,columns b]
  dtype : out.metadata.dtype=.floating .fp32
  device : out.metadata.device=a.metadata.device
  contiguous : out.metadata.isContiguous=Bool.true
  noNeg : out.metadata.isNeg=Bool.false
  separated : out.storage≠a.storage ∧ out.storage≠b.storage ∧ out.storage≠scale.storage
  alignment : Aligned out

structure InputContract (a b scale : TensorInfo) (out : Option TensorInfo) : Prop where
  rank : a.metadata.shape.length=2 ∧ b.metadata.shape.length=2 ∧ scale.metadata.shape.length=1
  layout : a.metadata.layout=.strided ∧ b.metadata.layout=.strided ∧ scale.metadata.layout=.strided
  dtype : a.metadata.dtype=.floating .fp32 ∧ b.metadata.dtype=.other "int8" ∧ scale.metadata.dtype=.floating .fp32
  device : a.metadata.device.kind="cuda" ∧ a.metadata.device=b.metadata.device ∧ a.metadata.device=scale.metadata.device
  contiguous : a.metadata.isContiguous=Bool.true ∧ b.metadata.isContiguous=Bool.true ∧ scale.metadata.isContiguous=Bool.true
  noNeg : a.metadata.isNeg=Bool.false ∧ b.metadata.isNeg=Bool.false ∧ scale.metadata.isNeg=Bool.false
  inputAlias : a.storage≠b.storage ∧ scale.storage≠b.storage
  shape : columns a=rows b ∧ scale.metadata.shape=[columns b]
  extent : a.metadata.numel<2^61 ∧ b.metadata.numel<2^61 ∧ rows a*columns b<2^61
  launch : columns a=0 ∨ columns b=0 ∨ (columns a<2^31 ∧ columns b≤65535*32)
  alignment : Aligned a ∧ Aligned scale
  output : ∀ provided,out=some provided → OutputContract a b scale provided

inductive InputError where
  | tensors | rank | layout | dtype | device | direct | shape | extent | launch | alignment
  | inputAlias | outputTensor | outputDescriptor | outputDirect | outputAlias | outputAlignment
  deriving DecidableEq

def checkOutput (a b scale out : TensorInfo) : Except InputError Unit :=
  if out.metadata.layout  !=  .strided || out.metadata.shape  !=  [rows a,columns b] ||
    out.metadata.dtype  !=  .floating .fp32 || out.metadata.device  !=  a.metadata.device then .error .outputDescriptor
  else if !out.metadata.isContiguous || out.metadata.isNeg then .error .outputDirect
  else if out.storage == a.storage || out.storage == b.storage || out.storage == scale.storage then .error .outputAlias
  else if out.metadata.numel>0 && out.metadata.dataPtr.toNat%4 != 0 then .error .outputAlignment
  else .ok ()

def checkTensors (a b scale : TensorInfo) (out : Option TensorInfo) : Except InputError Unit := do
  if a.metadata.shape.length  !=  2 || b.metadata.shape.length  !=  2 || scale.metadata.shape.length  !=  1 then throw .rank
  if a.metadata.layout  !=  .strided || b.metadata.layout  !=  .strided || scale.metadata.layout  !=  .strided then throw .layout
  if a.metadata.dtype  !=  .floating .fp32 || b.metadata.dtype  !=  .other "int8" || scale.metadata.dtype  !=  .floating .fp32 then throw .dtype
  if !a.metadata.isCuda || a.metadata.device  !=  b.metadata.device || a.metadata.device  !=  scale.metadata.device then throw .device
  if !a.metadata.isContiguous || a.metadata.isNeg || !b.metadata.isContiguous || b.metadata.isNeg ||
    !scale.metadata.isContiguous || scale.metadata.isNeg then throw .direct
  if a.storage == b.storage || scale.storage == b.storage then throw .inputAlias
  if columns a  !=  rows b || scale.metadata.shape  !=  [columns b] then throw .shape
  if a.metadata.numel≥2^61 || b.metadata.numel≥2^61 || rows a*columns b≥2^61 then throw .extent
  if columns a != 0 && columns b != 0 && (columns a≥2^31 || columns b>65535*32) then throw .launch
  if (a.metadata.numel>0 && a.metadata.dataPtr.toNat%4 != 0) ||
    (scale.metadata.numel>0 && scale.metadata.dataPtr.toNat%4 != 0) then throw .alignment
  match out with
  | none=>return ()
  | some provided=>checkOutput a b scale provided

inductive Argument where
  | tensor (info : TensorInfo)
  | nonTensor
inductive OutputArgument where
  | none
  | tensor (info : TensorInfo)
  | nonTensor

def checkInputs (a b scale : Argument) (out : OutputArgument) : Except InputError Unit :=
  match a,b,scale,out with
  | .tensor a,.tensor b,.tensor s,.none=>checkTensors a b s none
  | .tensor a,.tensor b,.tensor s,.tensor o=>checkTensors a b s (some o)
  | .tensor _,.tensor _,.tensor _,.nonTensor=>.error .outputTensor
  | _,_,_,_=>.error .tensors

def AdmittedArguments (a b scale : Argument) (out : OutputArgument) : Prop :=
  match a,b,scale,out with
  | .tensor a,.tensor b,.tensor s,.none=>InputContract a b s none
  | .tensor a,.tensor b,.tensor s,.tensor o=>InputContract a b s (some o)
  | _,_,_,_=>False
end VeriTile.Bench.Repairs.DequantizeMatmul.Direct

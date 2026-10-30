import VeriTile.Torch.Tensor
namespace VeriTile.Bench.Repairs.Int8DequantMatmul.Direct
open VeriTile.Torch
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
structure TensorInfo where
  metadata : TensorMetadata
  storage : Nat
  offset : Nat

def rows (x : TensorInfo) := x.metadata.shape[0]?.getD 0
def cols (x : TensorInfo) := x.metadata.shape[1]?.getD 0
def Aligned (x : TensorInfo) : Prop := 0 < x.metadata.numel → x.metadata.dataPtr.toNat%4=0
instance (x : TensorInfo) : Decidable (Aligned x) := inferInstanceAs (Decidable (_ → _))
def biasOffset (bias : Option TensorInfo) : Nat := (bias.map TensorInfo.offset).getD 0

structure BiasContract (a b : TensorInfo) (bias : TensorInfo) : Prop where
  descriptor : bias.metadata.layout=.strided ∧ bias.metadata.dtype=.floating .fp32 ∧
    bias.metadata.shape=[cols b] ∧ bias.metadata.device=a.metadata.device
  direct : bias.metadata.isContiguous=Bool.true ∧ bias.metadata.isNeg=Bool.false
  separated : bias.storage≠a.storage ∧ bias.storage≠b.storage
  aligned : Aligned bias

/-- Checked envelopes cover every autotune configuration, including padded
lanes and the final unused pointer increment. No integer-dot overflow bound. -/
structure LaunchContract (a b x w : TensorInfo) (bias : Option TensorInfo) : Prop where
  dimensions : rows a<2^61 ∧ cols b<2^61 ∧ cols a<2^61
  grid : ((rows a+15)/16)*((cols b+31)/32)≤2^31-1
  aFormed : a.offset+rows a*cols a+cols a+4095<2^63
  bFormed : b.offset+(cols a+4095)*cols b+cols b<2^63
  cFormed : 2*((rows a+255)*cols b+cols b+255)<2^63
  xFormed : 4*(x.offset+rows a)<2^63
  wFormed : 4*(w.offset+cols b)<2^63
  biasFormed : 4*(biasOffset bias+cols b+255)<2^63

structure InputContract (a b x w : TensorInfo) (bias : Option TensorInfo) : Prop where
  rank : a.metadata.shape.length=2 ∧ b.metadata.shape.length=2 ∧ x.metadata.shape.length=1 ∧ w.metadata.shape.length=1
  layout : a.metadata.layout=.strided ∧ b.metadata.layout=.strided ∧ x.metadata.layout=.strided ∧ w.metadata.layout=.strided
  dtype : a.metadata.dtype=.other "int8" ∧ b.metadata.dtype=.other "int8" ∧ x.metadata.dtype=.floating .fp32 ∧ w.metadata.dtype=.floating .fp32
  device : a.metadata.device.kind="cuda" ∧ b.metadata.device=a.metadata.device ∧ x.metadata.device=a.metadata.device ∧ w.metadata.device=a.metadata.device
  contiguous : a.metadata.isContiguous=Bool.true ∧ b.metadata.isContiguous=Bool.true ∧ x.metadata.isContiguous=Bool.true ∧ w.metadata.isContiguous=Bool.true
  noNeg : a.metadata.isNeg=Bool.false ∧ b.metadata.isNeg=Bool.false ∧ x.metadata.isNeg=Bool.false ∧ w.metadata.isNeg=Bool.false
  shape : rows b=cols a ∧ x.metadata.shape=[rows a] ∧ w.metadata.shape=[cols b]
  biasContract : ∀ t,bias=some t → BiasContract a b t
  inputAlias : x.storage≠a.storage ∧ x.storage≠b.storage ∧ w.storage≠a.storage ∧ w.storage≠b.storage
  alignment : Aligned x ∧ Aligned w
  launch : rows a=0 ∨ cols b=0 ∨ LaunchContract a b x w bias

inductive InputError where
  | tensors | biasTensor | rank | layout | dtype | device | direct | shape | biasDescriptor
  | biasDirect | inputAlias | alignment | dimensions | grid | formed
  deriving DecidableEq

def checkBias (a b bias : TensorInfo) : Except InputError Unit := do
  if bias.metadata.layout != .strided || bias.metadata.dtype != .floating .fp32 ||
    bias.metadata.shape != [cols b] || bias.metadata.device != a.metadata.device then throw .biasDescriptor
  if !bias.metadata.isContiguous || bias.metadata.isNeg then throw .biasDirect
  if bias.storage == a.storage || bias.storage == b.storage then throw .inputAlias
  if bias.metadata.numel>0 && bias.metadata.dataPtr.toNat%4 != 0 then throw .alignment
  return ()

def checkLaunch (a b x w : TensorInfo) (bias : Option TensorInfo) : Except InputError Unit := do
  if rows a≥2^61 || cols b≥2^61 || cols a≥2^61 then throw .dimensions
  if ((rows a+15)/16)*((cols b+31)/32)>2^31-1 then throw .grid
  if a.offset+rows a*cols a+cols a+4095≥2^63 || b.offset+(cols a+4095)*cols b+cols b≥2^63 ||
    2*((rows a+255)*cols b+cols b+255)≥2^63 || 4*(x.offset+rows a)≥2^63 ||
    4*(w.offset+cols b)≥2^63 || 4*(biasOffset bias+cols b+255)≥2^63 then throw .formed
  return ()

def checkTensors (a b x w : TensorInfo) (bias : Option TensorInfo) : Except InputError Unit := do
  if a.metadata.shape.length != 2 || b.metadata.shape.length != 2 || x.metadata.shape.length != 1 || w.metadata.shape.length != 1 then throw .rank
  if a.metadata.layout != .strided || b.metadata.layout != .strided || x.metadata.layout != .strided || w.metadata.layout != .strided then throw .layout
  if a.metadata.dtype != .other "int8" || b.metadata.dtype != .other "int8" || x.metadata.dtype != .floating .fp32 || w.metadata.dtype != .floating .fp32 then throw .dtype
  if !a.metadata.isCuda || b.metadata.device != a.metadata.device || x.metadata.device != a.metadata.device || w.metadata.device != a.metadata.device then throw .device
  if !a.metadata.isContiguous || a.metadata.isNeg || !b.metadata.isContiguous || b.metadata.isNeg ||
    !x.metadata.isContiguous || x.metadata.isNeg || !w.metadata.isContiguous || w.metadata.isNeg then throw .direct
  if rows b != cols a || x.metadata.shape != [rows a] || w.metadata.shape != [cols b] then throw .shape
  match bias with
  | some t=>checkBias a b t
  | none=>pure ()
  if x.storage == a.storage || x.storage == b.storage || w.storage == a.storage || w.storage == b.storage then throw .inputAlias
  if (x.metadata.numel>0 && x.metadata.dataPtr.toNat%4 != 0) || (w.metadata.numel>0 && w.metadata.dataPtr.toNat%4 != 0) then throw .alignment
  if rows a=0 || cols b=0 then pure () else checkLaunch a b x w bias

inductive Argument where
  | tensor (info : TensorInfo)
  | nonTensor
inductive BiasArgument where
  | none
  | tensor (info : TensorInfo)
  | nonTensor

def checkInputs (a b x w : Argument) (bias : BiasArgument) : Except InputError Unit :=
  match a,b,x,w,bias with
  | .tensor a,.tensor b,.tensor x,.tensor w,.none=>checkTensors a b x w none
  | .tensor a,.tensor b,.tensor x,.tensor w,.tensor t=>checkTensors a b x w (some t)
  | .tensor _,.tensor _,.tensor _,.tensor _,.nonTensor=>.error .biasTensor
  | _,_,_,_,_=>.error .tensors

def AdmittedArguments (a b x w : Argument) (bias : BiasArgument) : Prop :=
  match a,b,x,w,bias with
  | .tensor a,.tensor b,.tensor x,.tensor w,.none=>InputContract a b x w none
  | .tensor a,.tensor b,.tensor x,.tensor w,.tensor t=>InputContract a b x w (some t)
  | _,_,_,_,_=>False
end VeriTile.Bench.Repairs.Int8DequantMatmul.Direct

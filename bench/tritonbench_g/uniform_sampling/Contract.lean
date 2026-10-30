import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.UniformSampling
open VeriTile.Torch

structure GeneratorMetadata where
  object : Nat
  device : Device

inductive GeneratorArgument where
  | absent
  | supplied (data : GeneratorMetadata)
  | invalid

def generatorCheck (device : Device) : GeneratorArgument → Bool
  | .absent => Bool.true
  | .supplied data => data.device==device
  | .invalid => Bool.false

structure InputContract (output : TensorMetadata) (generator : GeneratorArgument)
    (lowerIsFloat upperIsFloat : Bool) : Prop where
  layout : output.layout=.strided
  dtype : output.dtype=.floating .fp32
  cuda : output.device.kind="cuda"
  contiguous : output.isContiguous=Bool.true
  noNeg : output.isNeg=Bool.false
  alignment : 0<output.numel → output.dataPtr.toNat%4=0
  extent : output.numel<2^63
  generator : generatorCheck output.device generator=Bool.true
  lowerType : lowerIsFloat=Bool.true
  upperType : upperIsFloat=Bool.true

inductive InputError where
  | tensor | metadata | extent | generator | bounds
  deriving DecidableEq, Repr

def checkInput (output : TensorMetadata) (generator : GeneratorArgument)
    (lowerIsFloat upperIsFloat : Bool) : Except InputError Unit :=
  if output.layout != .strided || output.dtype != .floating .fp32 || !output.isCuda ||
      !output.isContiguous || output.isNeg ||
      (output.numel>0 && output.dataPtr.toNat%4!=0) then .error .metadata
  else if output.numel>=2^63 then .error .extent
  else if !generatorCheck output.device generator then .error .generator
  else if !lowerIsFloat || !upperIsFloat then .error .bounds
  else .ok ()

def classifyTensor : ArgumentMetadata → Except InputError TensorMetadata
  | .nonTensor => .error .tensor
  | .tensor output => .ok output
end VeriTile.Bench.Repairs.UniformSampling

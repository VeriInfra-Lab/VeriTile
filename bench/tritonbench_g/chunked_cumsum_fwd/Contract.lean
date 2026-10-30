import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd
open VeriTile.Torch

def tensors (dt rate : TensorMetadata) (bias : Option TensorMetadata) := [dt,rate]++bias.toList

structure TensorContract (device : Device) (x : TensorMetadata) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
  device : x.device=device
  noNeg : x.isNeg=Bool.false
  alignment : 0<x.numel → x.dataPtr.toNat%4=0

def tensorCheck (device : Device) (x : TensorMetadata) : Bool :=
  x.layout==.strided && x.dtype==.floating .fp32 && x.isCuda && x.device==device && !x.isNeg &&
  (!(x.numel>0) || x.dataPtr.toNat%4==0)

structure InputContract (dt rate : TensorMetadata) (bias : Option TensorMetadata)
    (chunk : Int) (integerChunk booleanSoftplus floatLimits : Bool) : Prop where
  tensors : ∀ x∈tensors dt rate bias, TensorContract dt.device x
  rank : dt.shape.length=3
  rateShape : rate.shape=[dt.shape.getD 2 0]
  biasShape : ∀ x∈bias.toList, x.shape=[dt.shape.getD 2 0]
  chunkType : integerChunk=Bool.true
  chunkPositive : 0<chunk
  chunkBound : chunk<2^63
  chunkBlock : chunk≤1048576
  softplusType : booleanSoftplus=Bool.true
  limitsType : floatLimits=Bool.true
  shapeBounds : ∀ n∈dt.shape, n<2^63
  grid : 0<dt.numel → dt.shape.getD 0 0<2^31 ∧
    (dt.shape.getD 1 0+chunk.toNat-1)/chunk.toNat≤65535 ∧ dt.shape.getD 2 0≤65535

inductive InputError where
  | tensor | metadata | shape | chunk | softplus | limits | extent | grid
  deriving DecidableEq, Repr

def checkInput (dt rate : TensorMetadata) (bias : Option TensorMetadata)
    (chunk : Int) (integerChunk booleanSoftplus floatLimits : Bool) : Except InputError Unit :=
  if !(tensors dt rate bias).all (tensorCheck dt.device) then .error .metadata
  else if dt.shape.length != 3 || rate.shape != [dt.shape.getD 2 0] ||
    !(bias.toList.all (fun x=>x.shape==[dt.shape.getD 2 0])) then .error .shape
  else if !integerChunk || chunk<=0 || chunk>=2^63 || chunk>1048576 then .error .chunk
  else if !booleanSoftplus then .error .softplus
  else if !floatLimits then .error .limits
  else if !(dt.shape.all (fun n=>n<2^63)) then .error .extent
  else if dt.numel>0 && (dt.shape.getD 0 0>=2^31 ||
    (dt.shape.getD 1 0+chunk.toNat-1)/chunk.toNat>65535 || dt.shape.getD 2 0>65535) then .error .grid
  else .ok ()

def classifyTensor : ArgumentMetadata → Except InputError TensorMetadata
  | .nonTensor => .error .tensor
  | .tensor t => .ok t
end VeriTile.Bench.Repairs.ChunkedCumsumFwd

import VeriTile.Torch.Tensor
import Mathlib.Data.Int.Basic

namespace VeriTile.Bench.Repairs.RbeTritonTransform.Contract
open VeriTile.Torch

structure Position where
  integer : Bool
  value : Int

def paddedRows (x : TensorMetadata) : Nat := 2*((x.shape[1]?.getD 0+1)/2)

structure InputContract (x : TensorMetadata) (pos : Position) : Prop where
  layout : x.layout=.strided
  dtype : x.dtype=.floating .fp32
  cuda : x.device.kind="cuda"
  rank : x.shape.length=3
  contiguous : x.isContiguous=Bool.true
  resolved : x.isNeg=Bool.false
  aligned : 0<x.numel → x.dataPtr.toNat%4=0
  integer : pos.integer=Bool.true
  lower : -(2^63 : Int)≤pos.value
  upper : pos.value<(2^63 : Int)
  padded : 0<x.numel → pos.value+(paddedRows x : Int)-1<(2^63 : Int)

inductive Error where
  | tensor | dtype | cuda | shape | alignment | position
  deriving DecidableEq, Repr

def checkInputs (x : TensorMetadata) (pos : Position) : Except Error Unit :=
  if x.layout != .strided || x.dtype != .floating .fp32 then .error .dtype
  else if x.device.kind != "cuda" then .error .cuda
  else if x.shape.length != 3 || !x.isContiguous || x.isNeg then .error .shape
  else if x.numel>0 && x.dataPtr.toNat%4 != 0 then .error .alignment
  else if !pos.integer || pos.value< -(2^63 : Int) || pos.value≥(2^63 : Int) then .error .position
  else if x.numel>0 && pos.value+(paddedRows x : Int)-1≥(2^63 : Int) then .error .position
  else .ok ()

def checkArguments (x : ArgumentMetadata) (pos : Position) : Except Error Unit :=
  match x with
  | .nonTensor => .error .tensor
  | .tensor x => checkInputs x pos

def AdmittedArguments (x : ArgumentMetadata) (pos : Position) : Prop :=
  match x with
  | .nonTensor => False
  | .tensor x => InputContract x pos

end VeriTile.Bench.Repairs.RbeTritonTransform.Contract

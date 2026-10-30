import VeriTile.Triton.Float.BitValue
import Mathlib.Algebra.BigOperators.Group.List.Defs

/-! Metadata queried through the PyTorch API. These definitions describe API
observations, not the implementation of PyTorch or a kernel admission policy.
Only existing bit-value formats receive a distinguished dtype constructor;
other canonical PyTorch dtype/layout names remain opaque tags. -/
namespace VeriTile.Torch

inductive DType where
  | floating (format : Triton.FP.Format)
  | other (name : String)
  deriving DecidableEq, Repr

inductive Layout where
  | strided
  | other (name : String)
  deriving DecidableEq, Repr

/-- A resolved Tensor device; `kind` is the canonical PyTorch backend name. -/
structure Device where
  kind : String
  index : Nat
  deriving DecidableEq, Repr

structure TensorMetadata where
  dtype : DType
  layout : Layout
  device : Device
  shape : List Nat
  isContiguous : Bool
  isNeg : Bool
  /-- The data_ptr query on an admitted strided Tensor. Admission checks can
  reject unsupported metadata before inspecting this pointer observation. -/
  dataPtr : BitVec 64
  deriving DecidableEq, Repr

namespace TensorMetadata

def numel (t : TensorMetadata) : Nat := t.shape.prod
def isFP32 (t : TensorMetadata) : Bool := decide (t.dtype = .floating .fp32)
def isStrided (t : TensorMetadata) : Bool := decide (t.layout = .strided)
def isCuda (t : TensorMetadata) : Bool := decide (t.device.kind = "cuda")

end TensorMetadata

/-- Dynamic argument classification is separate from Tensor metadata. -/
inductive ArgumentMetadata where
  | nonTensor
  | tensor (metadata : TensorMetadata)
  deriving DecidableEq, Repr

instance : Coe TensorMetadata ArgumentMetadata := ⟨ArgumentMetadata.tensor⟩

end VeriTile.Torch

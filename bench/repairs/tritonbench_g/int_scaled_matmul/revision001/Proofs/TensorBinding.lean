import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.Contract
import bench.repairs.tritonbench_g.int_scaled_matmul.revision001.Proofs.FiniteIndex

namespace VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.TensorBinding
open VeriTile.Triton VeriTile.Torch
noncomputable section

inductive Kind where
  | int8 | int32 | fp32
def Kind.dtype : Kind → DType
  | .int8 => .other "int8"
  | .int32 => .other "int32"
  | .fp32 => .floating .fp32
def Kind.bytes : Kind → Nat
  | .int8 => 1
  | _ => 4

/-- Original Tensor identity and storage/view observations. No cloned or
materialized input supplies the values in the public references. -/
structure Tensor where
  object : Nat
  metadata : TensorMetadata
  (storageId storageBase storageElements storageOffset : Nat)
  physicalIndex : Nat → Nat

/-- Ordinary Torch API laws. Source-specific index/byte envelopes are absent:
they are checked explicitly in Contract.InputContract. -/
structure Laws (kind : Kind) (t : Tensor) : Prop where
  dtype : t.metadata.dtype=kind.dtype
  layout : t.metadata.layout=.strided
  bounds : ∀ i, i<t.metadata.numel → t.physicalIndex i<t.storageElements
  pointer : 0<t.metadata.numel →
    t.metadata.dataPtr.toNat=t.storageBase+kind.bytes*t.storageOffset
  contiguous : t.metadata.isContiguous=Bool.true →
    ∀ i, i<t.metadata.numel → t.physicalIndex i=t.storageOffset+i

abbrev Read := Nat → BitVec 8
def read32 (read : Read) (address : Nat) : BitVec 32 := BitVec.ofNat 32
  ((read address).toNat+256*(read (address+1)).toNat+
    65536*(read (address+2)).toNat+16777216*(read (address+3)).toNat)
def readFP32 (read : Read) (address : Nat) : FP.Value .fp32 := ⟨read32 read address⟩
def physicalAddress (kind : Kind) (t : Tensor) (i : Nat) := t.storageBase+kind.bytes*t.physicalIndex i

def int8Value (read : Read) (t : Tensor) (_laws : Laws .int8 t) (i : Nat) : Int :=
  let raw := read (physicalAddress .int8 t i)
  if t.metadata.isNeg then (-raw).toInt else raw.toInt
def fp32Value (read : Read) (t : Tensor) (_laws : Laws .fp32 t) (i : Nat) : FP.Value .fp32 :=
  let raw := readFP32 read (physicalAddress .fp32 t i)
  if t.metadata.isNeg then raw.neg else raw

theorem pointer_index (kind : Kind) (t : Tensor) (laws : Laws kind t)
    (contiguous : t.metadata.isContiguous=Bool.true) (i : Nat) (hi : i<t.metadata.numel) :
    t.metadata.dataPtr.toNat+kind.bytes*i=physicalAddress kind t i := by
  rw [laws.pointer (by omega)]
  simp only [physicalAddress,laws.contiguous contiguous i hi,Nat.mul_add,Nat.add_assoc]

theorem direct_int8 (read : Read) (t : Tensor) (laws : Laws .int8 t)
    (contiguous : t.metadata.isContiguous=Bool.true) (resolved : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<t.metadata.numel) :
    (read (t.metadata.dataPtr.toNat+i)).toInt=int8Value read t laws i := by
  have ptr := pointer_index .int8 t laws contiguous i hi
  simp only [Kind.bytes,Nat.one_mul] at ptr
  simp [int8Value,resolved,ptr]

theorem direct_fp32 (read : Read) (t : Tensor) (laws : Laws .fp32 t)
    (contiguous : t.metadata.isContiguous=Bool.true) (resolved : t.metadata.isNeg=Bool.false)
    (i : Nat) (hi : i<t.metadata.numel) :
    readFP32 read (t.metadata.dataPtr.toNat+4*i)=fp32Value read t laws i := by
  have ptr := pointer_index .fp32 t laws contiguous i hi
  simp only [Kind.bytes] at ptr
  simp [fp32Value,resolved,ptr]

def inputA : RegionName := "A"
def inputB : RegionName := "B"
def outputC : RegionName := "C"
def scales : RegionName := "S"

/-- The named source-pointer views are projections of one physical byte
reader. Overlapping read-only tensors therefore keep their actual values. -/
structure Inputs (read : Read) (a b scale : Tensor) (scaled : Bool)
    (s : FP.Structural.State (FP.Value .fp32)) : Prop where
  left : ∀ i, i<a.metadata.numel → s.mem inputA i=
    FP.Structural.Cell.mk .int (read (a.metadata.dataPtr.toNat+i)).toInt
  right : ∀ i, i<b.metadata.numel → s.mem inputB i=
    FP.Structural.Cell.mk .int (read (b.metadata.dataPtr.toNat+i)).toInt
  scale : scaled=Bool.true → ∀ i, i<scale.metadata.numel → s.mem scales i=
    FP.Structural.Cell.mk .real (readFP32 read (scale.metadata.dataPtr.toNat+4*i))

theorem separate_bytes (a c : TensorMetadata) (bytes : Nat) (positive : 0<bytes)
    (separate : Contract.Separate a bytes c) (i j ai cj : Nat)
    (hi : i<a.numel) (hj : j<c.numel) (hai : ai<bytes) (hcj : cj<4) :
    a.dataPtr.toNat+bytes*i+ai ≠ c.dataPtr.toNat+4*j+cj := by
  have ia := Nat.mul_le_mul_left bytes (show i+1≤a.numel by omega)
  have jc := Nat.mul_le_mul_left 4 (show j+1≤c.numel by omega)
  rw [Nat.mul_add,Nat.mul_one] at ia jc
  rcases separate with empty | empty | left | right <;> omega

#axiomsClean direct_int8
#axiomsClean direct_fp32
#axiomsClean separate_bytes
end
end VeriTile.Bench.Repairs.IntScaledMatmul.Revision001.TensorBinding

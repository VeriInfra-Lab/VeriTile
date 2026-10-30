import bench.tritonbench_g.quantize_kv_copy.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.QuantizeKvCopy
noncomputable section
abbrev Inputs := Role → VeriTile.Torch.TensorMetadata
abbrev MetadataWord := BitVec 32

def elementBytes : Role → Nat | .input => 4 | .destination => 4 | .output => 1 | .scale => 4
def base (m : Inputs) (r : Role) : Nat := (m r).dataPtr.toNat
def request (m : Inputs) (words : Nat → MetadataWord) : Request :=
  ⟨(m .destination).numel,(m .input).shape.getD 0 0,(m .input).shape.getD 1 0,
    (m .input).shape.getD 2 0 / 8,(m .output).shape.getD 0 0,1,fun s => (words s).toInt.toNat⟩
def Ranks (m : Inputs) : Prop := (m .input).shape.length=3 ∧ (m .destination).shape.length=1 ∧
  (m .output).shape.length=3 ∧ (m .scale).shape.length=3
def Types (m : Inputs) : Prop := (m .input).dtype=.floating .fp32 ∧
  (m .destination).dtype=.other "int32" ∧ (m .output).dtype=.other "int8" ∧ (m .scale).dtype=.floating .fp32

def Shapes (m : Inputs) (p : Request) : Prop := p.tokens≤p.sourceRows ∧
  (m .output).shape.drop 1=(m .input).shape.drop 1 ∧ (m .scale).shape=[p.capacity,p.headCount,p.groups]
def Separate (m : Inputs) : Prop := ∀ r s, r≠s →
  base m r+elementBytes r*(m r).numel≤base m s ∨ base m s+elementBytes s*(m s).numel≤base m r

def Bounds (p : Request) (words : Nat → MetadataWord) : Prop := ∀ s, s<p.tokens →
  0≤(words s).toInt ∧ (words s).toInt<p.capacity

def Unique (p : Request) : Prop := ∀ s t, s<p.tokens → t<p.tokens → p.slot s=p.slot t → s=t

def Width (m : Inputs) : Prop := 0<(m .input).shape.getD 1 0 ∧
  0<(m .input).shape.getD 2 0 ∧ (m .input).shape.getD 2 0 % 8=0

structure InputContract (m : Inputs) (words : Nat → MetadataWord) : Prop where
  layout : ∀ r, (m r).layout=.strided
  dtype : Types m
  cuda : ∀ r, (m r).device.kind="cuda"
  device : ∀ r, (m r).device=(m .input).device
  rank : Ranks m
  contiguous : ∀ r, (m r).isContiguous=Bool.true
  noNeg : ∀ r, (m r).isNeg=Bool.false
  width : Width m
  shapes : Shapes m (request m words)
  grid : (request m words).tokens<2^31 ∧ (request m words).headCount<2^16
  extent : ∀ r, (m r).numel<2^63
  tile : groupBlock (request m words)*8≤2^20
  alignment : ∀ r, 0<(m r).numel → base m r%elementBytes r=0
  separate : Separate m
  bounds : Bounds (request m words) words
  unique : Unique (request m words)

inductive InputError where
  | tensors | layout | dtype | cuda | device | rank | contiguous | negative | width
  | shapes | grid | extent | tile | alignment | overlap | bounds | unique
  deriving DecidableEq, Repr
open Classical in
def checkInputs (m : Inputs) (words : Nat → MetadataWord) : Except InputError Unit :=
  if ¬(∀ r, (m r).layout=.strided) then .error .layout
  else if ¬Types m then .error .dtype
  else if ¬(∀ r, (m r).device.kind="cuda") then .error .cuda
  else if ¬(∀ r, (m r).device=(m .input).device) then .error .device
  else if ¬Ranks m then .error .rank
  else if ¬(∀ r, (m r).isContiguous=Bool.true) then .error .contiguous
  else if ¬(∀ r, (m r).isNeg=Bool.false) then .error .negative
  else if ¬(Width m) then .error .width
  else if ¬Shapes m (request m words) then .error .shapes
  else if ¬((request m words).tokens<2^31 ∧ (request m words).headCount<2^16) then .error .grid
  else if ¬(∀ r, (m r).numel<2^63) then .error .extent
  else if ¬groupBlock (request m words)*8≤2^20 then .error .tile
  else if ¬(∀ r, 0<(m r).numel → base m r%elementBytes r=0) then .error .alignment
  else if ¬Separate m then .error .overlap
  else if ¬Bounds (request m words) words then .error .bounds
  else if ¬Unique (request m words) then .error .unique
  else .ok ()

def checkArguments (m : Option Inputs) (words : Nat → MetadataWord) : Except InputError Unit :=
  match m with | none => .error .tensors | some m => checkInputs m words

def Admitted (m : Option Inputs) (words : Nat → MetadataWord) : Prop :=
  match m with | none => False | some m => InputContract m words

/-- A direct data span of each caller Tensor; no allocator or snapshot appears. -/
def directLayout (m : Inputs) (objects : Role → Nat) : Layout :=
  ⟨fun r => ⟨objects r,base m r,0,elementBytes r*(m r).numel⟩⟩

/-- Ordinary address-space capacity and readback of initial signed-INT32 metadata. -/
structure TensorBinding (m : Inputs) (words : Nat → MetadataWord) (heap : Heap) : Prop where
  capacity : ∀ r, base m r+elementBytes r*(m r).numel≤2^64
  metadata : ∀ s, s<(request m words).tokens →
    loadDest heap (base m .destination+4*s)=(words s).toInt.toNat
end
end VeriTile.Bench.Repairs.QuantizeKvCopy

import bench.tritonbench_g.destindex_copy_kv1.Kernels
import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.DestindexCopyKv1
noncomputable section

/-- Independent row-scatter target, with the initial values of all source rows. -/
def RowCopy {α : Type} (c : Config) (dest : Nat → Nat)
    (source before after : Nat → α) : Prop :=
  (∀ row head dim, row<c.count → head<c.heads → dim<c.dims →
    after (flat c (dest row) head dim)=source (flat c row head dim)) ∧
  (∀ x, (¬∃ row head dim, row<c.count ∧ head<c.heads ∧ dim<c.dims ∧
    x=flat c (dest row) head dim) → after x=before x)

inductive Role where | source | destinations | output
  deriving DecidableEq, Repr
abbrev Inputs := Role → VeriTile.Torch.TensorMetadata
abbrev Word := BitVec 32
abbrev Memory := Nat → Word

def config (m : Inputs) : Config :=
  ⟨(m .destinations).numel,(m .source).shape.getD 0 0,(m .output).shape.getD 0 0,
    (m .source).shape.getD 1 0,(m .source).shape.getD 2 0⟩
def base (m : Inputs) (r : Role) : Nat := (m r).dataPtr.toNat
def addr (base index : Nat) : Nat := base+4*index
def destination (words : Nat → Word) (p : Nat) : Nat := (words p).toInt.toNat

def Ranks (m : Inputs) : Prop := (m .source).shape.length=3 ∧
  (m .output).shape.length=3 ∧ (m .destinations).shape.length=1
def Types (m : Inputs) : Prop := (m .source).dtype=.floating .fp32 ∧
  (m .output).dtype=.floating .fp32 ∧ (m .destinations).dtype=.other "int32"
def Shapes (m : Inputs) : Prop :=
  (m .source).shape.drop 1=(m .output).shape.drop 1 ∧ (config m).count≤(config m).sourceRows
def Separate (m : Inputs) : Prop := ∀ r, r≠.output →
  base m .output+4*(m .output).numel≤base m r ∨ base m r+4*(m r).numel≤base m .output

def Bounds (m : Inputs) (words : Nat → Word) : Prop := ∀ p, p<(config m).count →
  0≤(words p).toInt ∧ (words p).toInt<(config m).rows
def Unique (m : Inputs) (words : Nat → Word) : Prop := ∀ p q,
  p<(config m).count → q<(config m).count → destination words p=destination words q → p=q

structure InputContract (m : Inputs) (words : Nat → Word) : Prop where
  layout : ∀ r, (m r).layout=.strided
  rank : Ranks m
  dtype : Types m
  cuda : ∀ r, (m r).device.kind="cuda"
  device : ∀ r, (m r).device=(m .source).device
  contiguous : ∀ r, (m r).isContiguous=Bool.true
  noNeg : ∀ r, (m r).isNeg=Bool.false
  shapes : Shapes m
  extent : ∀ r, (m r).numel<2^31
  tile : block (config m).heads * block (config m).dims≤2^20
  alignment : ∀ r, 0<(m r).numel → base m r%4=0
  separate : Separate m
  bounds : Bounds m words
  unique : Unique m words

inductive InputError where
  | tensors | layout | rank | dtype | cuda | device | contiguous | negative
  | shapes | extent | tile | alignment | overlap | bounds | unique
  deriving DecidableEq, Repr
open Classical in
def checkInputs (m : Inputs) (words : Nat → Word) : Except InputError Unit :=
  if ¬(∀ r, (m r).layout=.strided) then .error .layout
  else if ¬Ranks m then .error .rank
  else if ¬Types m then .error .dtype
  else if ¬(∀ r, (m r).device.kind="cuda") then .error .cuda
  else if ¬(∀ r, (m r).device=(m .source).device) then .error .device
  else if ¬(∀ r, (m r).isContiguous=Bool.true) then .error .contiguous
  else if ¬(∀ r, (m r).isNeg=Bool.false) then .error .negative
  else if ¬Shapes m then .error .shapes
  else if ¬(∀ r, (m r).numel<2^31) then .error .extent
  else if ¬block (config m).heads * block (config m).dims≤2^20 then .error .tile
  else if ¬(∀ r, 0<(m r).numel → base m r%4=0) then .error .alignment
  else if ¬Separate m then .error .overlap
  else if ¬Bounds m words then .error .bounds
  else if ¬Unique m words then .error .unique
  else .ok ()

def checkArguments (m : Option Inputs) (words : Nat → Word) : Except InputError Unit :=
  match m with | none => .error .tensors | some m => checkInputs m words

def Admitted (m : Option Inputs) (words : Nat → Word) : Prop :=
  match m with | none => False | some m => InputContract m words

/-- Ordinary backing capacity and actual signed-int32 metadata readback. -/
structure TensorBinding (m : Inputs) (words : Nat → Word) (before : Memory) : Prop where
  capacity : ∀ r, base m r+4*(m r).numel≤2^64
  metadata : ∀ p, p<(config m).count → before (addr (base m .destinations) p)=words p

end
end VeriTile.Bench.Repairs.DestindexCopyKv1

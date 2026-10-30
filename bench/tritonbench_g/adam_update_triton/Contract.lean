import VeriTile.Torch.Tensor

namespace VeriTile.Bench.Repairs.AdamUpdateTriton
abbrev TensorMetadata := VeriTile.Torch.TensorMetadata
abbrev ArgumentMetadata := VeriTile.Torch.ArgumentMetadata

def DisjointRanges (a b : TensorMetadata) : Prop :=
  a.dataPtr.toNat+4*a.numel≤b.dataPtr.toNat ∨ b.dataPtr.toNat+4*b.numel≤a.dataPtr.toNat
instance (a b : TensorMetadata) : Decidable (DisjointRanges a b) := inferInstanceAs (Decidable (_ ∨ _))

/-- Full logical spans are disjoint; offsets within shared backing may pass.
Empty P returns before pointer queries and scalar normalization. -/
structure InputContract (p g m : TensorMetadata) : Prop where
  layout : p.layout=.strided ∧ g.layout=.strided ∧ m.layout=.strided
  dtype : p.dtype=.floating .fp32 ∧ g.dtype=.floating .fp32 ∧ m.dtype=.floating .fp32
  cuda : p.device.kind="cuda" ∧ g.device.kind="cuda" ∧ m.device.kind="cuda"
  sameDevice : g.device=p.device ∧ m.device=p.device
  enough : p.numel≤g.numel ∧ p.numel≤m.numel
  contiguous : p.isContiguous=Bool.true ∧ g.isContiguous=Bool.true ∧ m.isContiguous=Bool.true
  noNeg : p.isNeg=Bool.false ∧ g.isNeg=Bool.false ∧ m.isNeg=Bool.false
  alignment : 0<p.numel → p.dataPtr.toNat%16=0 ∧ g.dataPtr.toNat%16=0 ∧ m.dataPtr.toNat%16=0
  separate : 0<p.numel → DisjointRanges p g ∧ DisjointRanges p m ∧ DisjointRanges g m

inductive InputError where
  | tensors | layout | dtype | cuda | device | short | contiguous | negative | alignment | overlap
  deriving DecidableEq, Repr

def checkTensorInputs (p g m : TensorMetadata) : Except InputError Unit :=
  if p.layout != .strided || g.layout != .strided || m.layout != .strided then .error .layout
  else if p.dtype != .floating .fp32 || g.dtype != .floating .fp32 || m.dtype != .floating .fp32 then .error .dtype
  else if !(p.isCuda && g.isCuda && m.isCuda) then .error .cuda
  else if g.device != p.device || m.device != p.device then .error .device
  else if g.numel<p.numel || m.numel<p.numel then .error .short
  else if !(p.isContiguous && g.isContiguous && m.isContiguous) then .error .contiguous
  else if p.isNeg || g.isNeg || m.isNeg then .error .negative
  else if p.numel=0 then .ok ()
  else if p.dataPtr.toNat%16 != 0 || g.dataPtr.toNat%16 != 0 || m.dataPtr.toNat%16 != 0 then .error .alignment
  else if !(decide (DisjointRanges p g) && decide (DisjointRanges p m) && decide (DisjointRanges g m)) then .error .overlap
  else .ok ()

def checkInputs (p g m : ArgumentMetadata) : Except InputError Unit :=
  match p,g,m with
  | .tensor p,.tensor g,.tensor m => checkTensorInputs p g m
  | _,_,_ => .error .tensors

def AdmittedArguments (p g m : ArgumentMetadata) : Prop :=
  match p,g,m with
  | .tensor p,.tensor g,.tensor m => InputContract p g m
  | _,_,_ => False
end VeriTile.Bench.Repairs.AdamUpdateTriton

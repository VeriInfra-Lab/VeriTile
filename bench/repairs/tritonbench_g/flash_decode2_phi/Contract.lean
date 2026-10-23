import VeriTile.Torch.FP32
import bench.repairs.tritonbench_g.square_matrix.Kernels
import bench.repairs.tritonbench_g.flash_decode2_phi.Proofs.Geometry

namespace VeriTile.Bench.Repairs.FlashDecode2.Checked

structure Observation where
  metadata : VeriTile.Torch.TensorMetadata
  storage : VeriTile.Torch.FP32.Storage
  storageOffset : Nat
  strides : List Nat
  object : Nat
  isConj : Bool

structure Call where
  (mid log lengths output : Observation)
  blockSeq : Nat
  masked : Bool

def dimension (c : Call) (n : Nat) := c.mid.metadata.shape[n]?.getD 0
def batches (c : Call) := dimension c 0
def heads (c : Call) := dimension c 1
def capacity (c : Call) := dimension c 2
def width (c : Call) := dimension c 3
def block (c : Call) := if c.masked then SquareMatrix.block (width c) else width c

def geometry (c : Call) : Geometry :=
  ⟨batches c,heads c,capacity c,width c,c.blockSeq,block c,c.masked⟩
def tensors (c : Call) := [c.mid,c.log,c.lengths,c.output]
def active (c : Call) : Prop := 0 < batches c ∧ 0 < heads c
instance (c : Call) : Decidable (active c) := inferInstanceAs (Decidable (_ ∧ _))

def metadataChecks (c : Call) : Prop :=
  (∀ t ∈ tensors c,t.metadata.layout = .strided ∧ t.metadata.device.kind = "cuda" ∧
    t.metadata.device = c.mid.metadata.device ∧ t.metadata.isContiguous = Bool.true ∧
    t.storageOffset = 0 ∧ t.metadata.isNeg = Bool.false ∧ t.isConj = Bool.false) ∧
  (∀ t ∈ [c.mid,c.log,c.output],t.metadata.dtype = .floating .fp32) ∧
  c.lengths.metadata.dtype = .other "int32" ∧
  c.mid.metadata.shape = [batches c,heads c,capacity c,width c] ∧
  c.log.metadata.shape = [batches c,heads c,capacity c] ∧
  c.lengths.metadata.shape = [batches c] ∧
  c.output.metadata.shape = [batches c,heads c,width c] ∧
  (geometry c).Valid ∧ (c.masked = Bool.false → width c ∈ ([16,32,64,128] : List Nat))
instance (c : Call) : Decidable (metadataChecks c) := by
  unfold metadataChecks Geometry.Valid; infer_instance

def apart (x y : Observation) : Prop :=
  x.storage.base+4*x.storage.elements ≤ y.storage.base ∨
  y.storage.base+4*y.storage.elements ≤ x.storage.base
instance (x y : Observation) : Decidable (apart x y) := inferInstanceAs (Decidable (_ ∨ _))

def launchChecks (c : Call) : Prop :=
  (∀ t ∈ tensors c,0 < t.metadata.numel → t.metadata.dataPtr.toNat % 4 = 0 ∧
    t.metadata.dataPtr.toNat+4*(t.metadata.numel+block c) ≤ 2^64) ∧
  (∀ t ∈ [c.mid,c.log,c.lengths],apart c.output t) ∧
  apart c.lengths c.mid ∧ apart c.lengths c.log
instance (c : Call) : Decidable (launchChecks c) := by unfold launchChecks; infer_instance

def contentChecks (c : Call) (lengths : Nat → Int) : Prop :=
  ∀ b ∈ List.range (batches c),0 < lengths b ∧ lengths b ≤ (capacity c*c.blockSeq : Nat) ∧
    lengths b ≤ 2^31-1
instance (c : Call) (lengths : Nat → Int) : Decidable (contentChecks c lengths) := by
  unfold contentChecks; infer_instance

structure InputContract (c : Call) (lengths : Nat → Int) : Prop where
  metadata : metadataChecks c
  launch : active c → launchChecks c
  contents : active c → contentChecks c lengths

def checkInput (c : Call) (lengths : Nat → Int) : Except String Unit :=
  if ¬metadataChecks c then .error "metadata"
  else if active c ∧ ¬launchChecks c then .error "pointer/storage"
  else if active c ∧ ¬contentChecks c lengths then .error "length contents"
  else .ok ()

inductive Argument where | nonTensor | tensor (observation : Observation)
inductive SequenceBlock where | nonInteger | integer (value : Int)

def checkArguments (masked : Bool) (m e l o : Argument) (q : SequenceBlock)
    (lengths : Nat → Int) : Except String Unit :=
  match m,e,l,o,q with
  | .tensor m,.tensor e,.tensor l,.tensor o,.integer q =>
    if 0 < q then checkInput ⟨m,e,l,o,q.toNat,masked⟩ lengths else .error "sequence block"
  | _,_,_,_,_ => .error "argument type"

def AdmittedArguments (masked : Bool) (m e l o : Argument) (q : SequenceBlock)
    (lengths : Nat → Int) : Prop :=
  match m,e,l,o,q with
  | .tensor m,.tensor e,.tensor l,.tensor o,.integer q =>
    0 < q ∧ InputContract ⟨m,e,l,o,q.toNat,masked⟩ lengths
  | _,_,_,_,_ => False

end VeriTile.Bench.Repairs.FlashDecode2.Checked

namespace VeriTile.Bench.Repairs.FlashDecode2Phi
abbrev Call := {c : FlashDecode2.Checked.Call // c.masked = Bool.true}
abbrev InputContract (c : Call) := FlashDecode2.Checked.InputContract c.val
abbrev checkInput (c : Call) := FlashDecode2.Checked.checkInput c.val
abbrev checkArguments := FlashDecode2.Checked.checkArguments Bool.true
end VeriTile.Bench.Repairs.FlashDecode2Phi

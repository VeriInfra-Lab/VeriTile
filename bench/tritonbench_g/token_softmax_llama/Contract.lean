import VeriTile.Torch.FP32
import bench.tritonbench_g.square_matrix.Contract

namespace VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked
open VeriTile.Triton
abbrev Metadata := VeriTile.Torch.TensorMetadata

/-- Observed Tensor API fields. Strides are nonnegative, as required by Torch's
strided Tensor constructor. Storage refers to the actual backing allocation. -/
structure Observation where
  metadata : Metadata
  strides : List Nat
  storage : VeriTile.Torch.FP32.Storage
  storageOffset : Nat
  object : Nat

def rows (x : Observation) := x.metadata.shape.headD 0
def cols (x : Observation) := x.metadata.shape[1]?.getD 0
def rs (x : Observation) := x.strides.headD 0
def cs (x : Observation) := x.strides[1]?.getD 0
abbrev block := VeriTile.Bench.Repairs.SquareMatrix.block

structure Call where
  (input start length output : Observation)
  maximum : Nat

def tensors (c : Call) := [c.input,c.start,c.length,c.output]
def active (c : Call) := 0 < rows c.start ∧ 0 < rows c.input ∧ 0 < cols c.input ∧ 0 < c.maximum
instance (c : Call) : Decidable (active c) := inferInstanceAs (Decidable (_ ∧ _))
def contentActive (c : Call) := 0 < rows c.start ∧ 0 < rows c.input ∧ 0 < cols c.input
instance (c : Call) : Decidable (contentActive c) := inferInstanceAs (Decidable (_ ∧ _))
def nonoverlap (H K R C : Nat) :=
  ((H ≤ 1 ∨ (K-1)*C < R) ∧ (K ≤ 1 ∨ 0 < C)) ∨
  ((K ≤ 1 ∨ (H-1)*R < C) ∧ (H ≤ 1 ∨ 0 < R))
instance (H K R C : Nat) : Decidable (nonoverlap H K R C) := inferInstanceAs (Decidable (_ ∨ _))
def formed (c : Call) (t : Observation) :=
  (rows c.input-1)*rs t+(cols c.input+block c.maximum-1)*cs t

def tensorChecks (c : Call) : Prop :=
  (∀ t∈tensors c,t.metadata.layout=.strided) ∧
  c.input.metadata.dtype=.floating .fp32 ∧ c.output.metadata.dtype=.floating .fp32 ∧
  c.start.metadata.dtype=.other "int32" ∧ c.length.metadata.dtype=.other "int32" ∧
  (∀ t∈tensors c,t.metadata.device.kind="cuda") ∧
  (∀ t∈tensors c,t.metadata.device=c.input.metadata.device) ∧
  c.input.metadata.shape.length=2 ∧ c.output.metadata.shape=c.input.metadata.shape ∧
  c.start.metadata.shape.length=1 ∧ c.length.metadata.shape=c.start.metadata.shape ∧
  (∀ t∈tensors c,t.metadata.isNeg=Bool.false)
instance (c : Call) : Decidable (tensorChecks c) := inferInstanceAs (Decidable (_ ∧ _))

def launchChecks (c : Call) : Prop :=
  rows c.start < 2^31 ∧ rows c.input < 2^16 ∧ block c.maximum ≤ 2^20 ∧
  nonoverlap (rows c.input) (cols c.input) (rs c.output) (cs c.output) ∧
  (∀ t∈tensors c,∀ s∈t.strides,s < 2^63) ∧
  (∀ t∈tensors c,t.metadata.dataPtr.toNat%4=0) ∧
  cols c.input+block c.maximum < 2^63 ∧
  formed c c.input < 2^63 ∧ formed c c.output < 2^63 ∧
  (rows c.start-1)*rs c.start < 2^63 ∧ (rows c.start-1)*rs c.length < 2^63 ∧
  c.input.metadata.dataPtr.toNat+4*formed c c.input+4 ≤ 2^64 ∧
  c.output.metadata.dataPtr.toNat+4*formed c c.output+4 ≤ 2^64 ∧
  c.start.metadata.dataPtr.toNat+4*((rows c.start-1)*rs c.start)+4 ≤ 2^64 ∧
  c.length.metadata.dataPtr.toNat+4*((rows c.start-1)*rs c.length)+4 ≤ 2^64 ∧
  (∀ t∈[c.input,c.start,c.length],
    c.output.storage.base+4*c.output.storage.elements ≤ t.storage.base ∨
    t.storage.base+4*t.storage.elements ≤ c.output.storage.base)
instance (c : Call) : Decidable (launchChecks c) := inferInstanceAs (Decidable (_ ∧ _))

/-- Shape, dtype, pointer and launch metadata checks. -/
structure MetadataContract (c : Call) : Prop where
  tensors : tensorChecks c
  launch : active c → launchChecks c

def check (c : Call) : Except String Unit :=
  if ¬tensorChecks c then .error "tensor metadata"
  else if active c ∧ ¬launchChecks c then .error "launch metadata"
  else .ok ()

inductive Argument where | nonTensor | tensor (observation : Observation)
inductive Maximum where | nonInteger | integer (value : Int)

def checkArguments : Argument → Argument → Argument → Argument → Maximum → Except String Unit
  | .tensor x,.tensor s,.tensor n,.tensor o,.integer k =>
    if 0 ≤ k then check ⟨x,s,n,o,k.toNat⟩ else .error "maximum"
  | _,_,_,_,_ => .error "argument type"

def ContentChecks (c : Call) (starts lengths : Nat → Int) : Prop :=
  (∀ b ∈ List.range (rows c.start),
    0 ≤ starts b ∧ 0 ≤ lengths b ∧ lengths b ≤ (c.maximum : Int) ∧
    starts b+lengths b ≤ (cols c.input : Int)) ∧
  (∀ b ∈ List.range (rows c.start), ∀ d ∈ List.range (rows c.start),
    b ≠ d → 0 < lengths b → 0 < lengths d →
    starts b+lengths b ≤ starts d ∨ starts d+lengths d ≤ starts b)

instance (c : Call) (s n : Nat → Int) : Decidable (ContentChecks c s n) := by
  unfold ContentChecks; infer_instance

/-- Full successful input contract, including values queried by int(tensor[b]).
Truly empty axes read no contents; zero maximum still validates zero lengths. -/
structure InputContract (c : Call) (starts lengths : Nat → Int) : Prop where
  metadata : MetadataContract c
  contents : contentActive c → ContentChecks c starts lengths

def checkContents (c : Call) (starts lengths : Nat → Int) : Bool :=
  decide (ContentChecks c starts lengths)

def checkInput (c : Call) (starts lengths : Nat → Int) : Except String Unit :=
  match check c with
  | .error e => .error e
  | .ok () => if contentActive c ∧ checkContents c starts lengths = Bool.false then
      .error "metadata contents" else .ok ()

def checkCallArguments (x s n o : Argument) (k : Maximum) (starts lengths : Nat → Int) : Except String Unit :=
  match x,s,n,o,k with
  | .tensor x,.tensor s,.tensor n,.tensor o,.integer k =>
    if 0 ≤ k then checkInput ⟨x,s,n,o,k.toNat⟩ starts lengths else .error "maximum"
  | _,_,_,_,_ => .error "argument type"

def AdmittedCallArguments (x s n o : Argument) (k : Maximum) (starts lengths : Nat → Int) : Prop :=
  match x,s,n,o,k with
  | .tensor x,.tensor s,.tensor n,.tensor o,.integer k =>
    0 ≤ k ∧ InputContract ⟨x,s,n,o,k.toNat⟩ starts lengths
  | _,_,_,_,_ => False

end VeriTile.Bench.Repairs.TokenSoftmaxLlama.Checked

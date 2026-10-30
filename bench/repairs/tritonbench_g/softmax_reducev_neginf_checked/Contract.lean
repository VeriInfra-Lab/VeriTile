import bench.repairs.tritonbench_g.token_softmax_llama.Contract
import Init.Data.Nat.Power2.Lemmas

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked
abbrev Observation := TokenSoftmaxLlama.Checked.Observation
abbrev Argument := TokenSoftmaxLlama.Checked.Argument
abbrev Maximum := TokenSoftmaxLlama.Checked.Maximum

def dim (x : Observation) (n : Nat) := x.metadata.shape[n]?.getD 0

structure Call where
  (logics values output locations starts lengths : Observation)
  maximum : Nat

def tensors (c : Call) := [c.logics,c.values,c.output,c.locations,c.starts,c.lengths]
def batch (c : Call) := dim c.lengths 0
def heads (c : Call) := dim c.logics 0
def dimensions (c : Call) := dim c.values 2
def active (c : Call) := 0 < batch c ∧ 0 < heads c
instance (c : Call) : Decidable (active c) := inferInstanceAs (Decidable (_ ∧ _))

def metadataChecks (c : Call) :=
  (∀ t∈tensors c,t.metadata.layout=.strided ∧ t.metadata.device.kind="cuda") ∧
  c.logics.metadata.shape.length=2 ∧ c.values.metadata.shape.length=3 ∧ c.output.metadata.shape.length=3 ∧
  c.locations.metadata.shape.length=2 ∧ c.starts.metadata.shape.length=1 ∧ c.lengths.metadata.shape.length=1 ∧
  (dimensions c).isPowerOfTwo ∧ 0 < c.maximum ∧
  dim c.values 1=heads c ∧ c.output.metadata.shape=[batch c,heads c,dimensions c] ∧
  c.starts.metadata.shape=c.lengths.metadata.shape ∧ c.locations.metadata.shape=[batch c,c.maximum] ∧
  (∀ t∈[c.logics,c.values,c.output],t.metadata.dtype=.floating .fp32) ∧
  (∀ t∈[c.locations,c.starts,c.lengths],t.metadata.dtype=.other "int32") ∧
  (∀ t∈tensors c,t.metadata.device=c.logics.metadata.device ∧
    t.metadata.isNeg=Bool.false ∧ t.metadata.isContiguous=Bool.true) ∧
  c.maximum ≤ 2^31-64 ∧ (∀ t∈tensors c,t.metadata.numel ≤ 2^31-64)
instance (c : Call) : Decidable (metadataChecks c) := inferInstanceAs (Decidable (_ ∧ _))

def launchChecks (c : Call) :=
  batch c < 2^31 ∧ heads c < 2^16 ∧ dimensions c*64 ≤ 2^20 ∧
  (∀ t∈tensors c,0 < t.metadata.numel → t.metadata.dataPtr.toNat%4=0) ∧
  (∀ t∈[c.logics,c.values,c.locations,c.starts,c.lengths],
    c.output.storage.base+4*c.output.storage.elements ≤ t.storage.base ∨
    t.storage.base+4*t.storage.elements ≤ c.output.storage.base)
instance (c : Call) : Decidable (launchChecks c) := inferInstanceAs (Decidable (_ ∧ _))

structure MetadataContract (c : Call) : Prop where
  metadata : metadataChecks c
  launch : active c → launchChecks c

def check (c : Call) : Except String Unit :=
  if ¬metadataChecks c then .error "metadata"
  else if active c ∧ ¬launchChecks c then .error "launch"
  else .ok ()

def checkArguments : Argument → Argument → Argument → Argument → Argument → Argument → Maximum → Except String Unit
  | .tensor x,.tensor v,.tensor o,.tensor p,.tensor s,.tensor n,.integer k =>
    if 0 < k then check ⟨x,v,o,p,s,n,k.toNat⟩ else .error "maximum"
  | _,_,_,_,_,_,_ => .error "argument type"

def ContentChecks (c : Call) (starts lengths pages : Nat → Int) : Prop :=
  (∀ b ∈ List.range (batch c),
    0 ≤ starts b ∧ 0 < lengths b ∧ lengths b ≤ (c.maximum : Int) ∧
    starts b+lengths b ≤ (dim c.logics 1 : Int)) ∧
  (∀ b ∈ List.range (batch c), ∀ k ∈ List.range (lengths b).toNat,
    let p := pages (b*c.maximum+(c.maximum-(lengths b).toNat)+k)
    0 ≤ p ∧ p < (dim c.values 0 : Int))

instance (c : Call) (s n p : Nat → Int) : Decidable (ContentChecks c s n p) := by
  unfold ContentChecks; infer_instance

structure InputContract (c : Call) (starts lengths pages : Nat → Int) : Prop where
  metadata : MetadataContract c
  contents : active c → ContentChecks c starts lengths pages

def checkContents (c : Call) (starts lengths pages : Nat → Int) : Bool :=
  decide (ContentChecks c starts lengths pages)

def checkInput (c : Call) (starts lengths pages : Nat → Int) : Except String Unit :=
  match check c with
  | .error e => .error e
  | .ok () => if active c ∧ checkContents c starts lengths pages = Bool.false then
      .error "metadata contents" else .ok ()

def checkCallArguments (x v o p s n : Argument) (k : Maximum)
    (starts lengths pages : Nat → Int) : Except String Unit :=
  match x,v,o,p,s,n,k with
  | .tensor x,.tensor v,.tensor o,.tensor p,.tensor s,.tensor n,.integer k =>
    if 0 < k then checkInput ⟨x,v,o,p,s,n,k.toNat⟩ starts lengths pages else .error "maximum"
  | _,_,_,_,_,_,_ => .error "argument type"

def AdmittedCallArguments (x v o p s n : Argument) (k : Maximum)
    (starts lengths pages : Nat → Int) : Prop :=
  match x,v,o,p,s,n,k with
  | .tensor x,.tensor v,.tensor o,.tensor p,.tensor s,.tensor n,.integer k =>
    0 < k ∧ InputContract ⟨x,v,o,p,s,n,k.toNat⟩ starts lengths pages
  | _,_,_,_,_,_,_ => False
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.Checked

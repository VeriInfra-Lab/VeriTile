import VeriTile.Triton.Float.LogicalProgramIO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.NativeProfile
import VeriTile.Triton.Float.ProgramIO
import bench.repairs.tritonbench_g.softmax_reducev_neginf_checked.Proofs.RealMath

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

/-- The padded logical extent is derived from arbitrary per-batch lengths.
It is not the source's `maximum`, and introduces no maximum-length guard. -/
def capacity {B : Nat} (counts : Fin B → Nat) : Nat := Finset.univ.sup counts
theorem count_le_capacity {B : Nat} (counts : Fin B → Nat) (b : Fin B) :
    counts b ≤ capacity counts := Finset.le_sup (Finset.mem_univ b)

def inputs (c : Host.Call) (counts : Fin c.batch → Nat) : List IOChannel :=
  [⟨.nat,[c.batch],"Lengths"⟩,
   ⟨.nat,[c.batch],"Starts"⟩,
   ⟨.int,[c.batch,capacity counts],"Locations"⟩,
   ⟨.fp32,[c.batch,c.heads,capacity counts],"Logics"⟩,
   ⟨.fp32,[c.batch,c.heads,capacity counts,c.dimensions],"V"⟩]
def outputs (c : Host.Call) : List IOChannel :=
  [⟨.fp32,[c.batch,c.heads,c.dimensions],"Out"⟩]
abbrev Inputs (A : Type) (c : Host.Call) (counts : Fin c.batch → Nat) :=
  ChannelValues A (inputs c counts)
abbrev Outputs (A : Type) (c : Host.Call) := ChannelValues A (outputs c)

def length {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs A c counts) (b : Fin c.batch) : Nat := xs ⟨0,by simp [inputs]⟩ (b,())
def start {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs A c counts) (b : Fin c.batch) : Nat := xs ⟨1,by simp [inputs]⟩ (b,())
def page {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs A c counts) (b : Fin c.batch) (k : Fin (capacity counts)) : Int :=
  xs ⟨2,by simp [inputs]⟩ (b,k,())
def logit {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs A c counts) (b : Fin c.batch) (h : Fin c.heads) (k : Fin (capacity counts)) : A :=
  xs ⟨3,by simp [inputs]⟩ (b,h,k,())
def value {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs A c counts) (b : Fin c.batch) (h : Fin c.heads)
    (k : Fin (capacity counts)) (d : Fin c.dimensions) : A :=
  xs ⟨4,by simp [inputs]⟩ (b,h,k,d,())

def LengthsMatch {c : Host.Call} {counts : Fin c.batch → Nat} (xs : Inputs A c counts) : Prop :=
  ∀ b, 0 < c.heads → length xs b=counts b

def read (c : Host.Call) (counts : Fin c.batch → Nat)
    (port : Fin (inputs c counts).length) (xs : Inputs A c counts) :
    TileIndex (inputs c counts)[port].shape → Nat :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ => fun j => j.1.val
  | ⟨2,_⟩ => fun j => j.1.val*c.maximum+(c.maximum-counts j.1)+j.2.1.val
  | ⟨3,_⟩ => fun j => j.2.1.val*c.tokens+(start xs j.1+j.2.2.1.val)
  | ⟨4,_⟩ => fun j => j.2.1.val*c.dimensions+j.2.2.2.1.val+
      (page xs j.1 j.2.2.1*(c.heads*c.dimensions : Nat)).toNat
  | ⟨k+5,hp⟩ => False.elim (by simp [inputs] at hp)

def inputMask (c : Host.Call) (counts : Fin c.batch → Nat)
    (port : Fin (inputs c counts).length) : TileIndex (inputs c counts)[port].shape → Prop :=
  match port with
  | ⟨0,_⟩ | ⟨1,_⟩ => fun _ => 0 < c.heads
  | ⟨2,_⟩ => fun j => 0 < c.heads ∧ j.2.1.val < counts j.1
  | ⟨3,_⟩ | ⟨4,_⟩ => fun j => j.2.2.1.val < counts j.1
  | ⟨k+5,hp⟩ => False.elim (by simp [inputs] at hp)

def memory (A : Type) (c : Host.Call) (counts : Fin c.batch → Nat) : MemoryIO A where
  inputs := inputs c counts
  outputs := outputs c
  scratch := []
  buffers := ["Lengths","Starts","Locations","Logics","V","Out"]
  read := read c counts
  inputMask := fun port _ => inputMask c counts port
  write := Fin.cases (fun _ j => (j.1.val*c.heads+j.2.1.val)*c.dimensions+j.2.2.1.val)
    (fun k => k.elim0)
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => k.elim0
  scratchMask := fun k => k.elim0

def invocation (M : Algebra A) (c : Host.Call) (j : Host.Job) : Invocation A where
  kernel := FPValue.source (Host.parameters c)
  backend := M
  clearRegisters := Bool.true
  pids := fun ax => if ax=0 then j.batch else if ax=1 then j.head else 0
  numPids := some (fun ax => if ax=0 then c.batch else if ax=1 then c.heads else 1)

/-- Complete actual-source launch order, with typed metadata and active
gathered windows. Neither input values nor memory are replaced by a callback. -/
def sourceIO (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
    (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
    (counts : Fin c.batch → Nat) (work : List Host.Job) : ProgramIO A B :=
  { memory A c counts with
    schedule := fun _ => work.map (fun j => invocation (M j) c j)
    encode := encode
    observe := Fin.cases (fun j => observe j.1 j.2.1 j.2.2.1) (fun k => k.elim0) }

def result {c : Host.Call} (f : Fin c.batch → Fin c.heads → Fin c.dimensions → A) : Outputs A c :=
  Fin.cases (fun j => f j.1 j.2.1 j.2.2.1) (fun k => k.elim0)

def logitAt [Inhabited A] {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs A c counts) (b : Fin c.batch) (h : Fin c.heads) (k : Nat) : A :=
  if hk : k < capacity counts then logit xs b h ⟨k,hk⟩ else default
def valueAt [Inhabited A] {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : Inputs A c counts) (b : Fin c.batch) (h : Fin c.heads) (k : Nat) (d : Fin c.dimensions) : A :=
  if hk : k < capacity counts then value xs b h ⟨k,hk⟩ d else default

def nativeModel (profiles : Host.Job → NativeGraph.Profile) (j : Host.Job) :=
  NativeGraph.algebra (NativeGraph.profilePlan (profiles j))

def nativeObserve (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (b : Fin c.batch) (h : Fin c.heads) (d : Fin c.dimensions) (v : NativeGraph.Node) :
    NativeGraph.Word :=
  NativeGraph.eval ops (v ((NativeGraph.profilePlan (profiles ⟨b.val,h.val⟩)).storeReplica c.dimensions d.val))

def nativeIO (profiles : Host.Job → NativeGraph.Profile) (ops : NativeGraph.Instructions)
    (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job) :
    ProgramIO NativeGraph.Word NativeGraph.Node :=
  sourceIO (nativeModel profiles) NativeGraph.atom c (nativeObserve profiles ops c) counts work



/-- Typed logical inputs retain the full padded capacity, not only live keys.
Lengths, starts and signed page values remain independent input channels. -/
structure LogicalInputs (A : Type) (c : Host.Call) (counts : Fin c.batch → Nat) where
  lengths : Fin c.batch → Nat
  starts : Fin c.batch → Nat
  pages : Fin c.batch → Fin (capacity counts) → Int
  logits : Fin c.batch → Fin c.heads → Fin (capacity counts) → A
  values : Fin c.batch → Fin c.heads → Fin (capacity counts) → Fin c.dimensions → A

def logicalInputs (A : Type) (c : Host.Call) (counts : Fin c.batch → Nat) :
    Inputs A c counts ≃ LogicalInputs A c counts where
  toFun := fun xs => ⟨length xs,start xs,page xs,logit xs,value xs⟩
  invFun := fun xs => Fin.cases (fun j => xs.lengths j.1)
    (Fin.cases (fun j => xs.starts j.1) (Fin.cases (fun j => xs.pages j.1 j.2.1)
      (Fin.cases (fun j => xs.logits j.1 j.2.1 j.2.2.1)
        (Fin.cases (fun j => xs.values j.1 j.2.1 j.2.2.1 j.2.2.2.1) (fun i => i.elim0)))))
  left_inv := by intro xs; funext port j; fin_cases port <;> rfl
  right_inv := by rintro ⟨l,s,p,q,v⟩; rfl

def logicalOutputs (A : Type) (c : Host.Call) : Outputs A c ≃
    (Fin c.batch → Fin c.heads → Fin c.dimensions → A) :=
  (ChannelValues.singleEquiv A _).trans (Values.tensor3Equiv A .fp32 _ _ _)

def LogicalInputs.logitAt [Inhabited A] {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : LogicalInputs A c counts) (b : Fin c.batch) (h : Fin c.heads) (k : Nat) : A :=
  if hk : k<capacity counts then xs.logits b h ⟨k,hk⟩ else default

def LogicalInputs.valueAt [Inhabited A] {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : LogicalInputs A c counts) (b : Fin c.batch) (h : Fin c.heads) (k : Nat)
    (d : Fin c.dimensions) : A :=
  if hk : k<capacity counts then xs.values b h ⟨k,hk⟩ d else default

def LogicalRealDomain {c : Host.Call} {counts : Fin c.batch → Nat}
    (xs : LogicalInputs (WithBot ℝ) c counts) : Prop :=
  (∀ b, 0<c.heads → xs.lengths b=counts b) ∧
  (∀ b h, ∃ k, k<counts b ∧ xs.logitAt b h k≠⊥) ∧
  (∀ b h k, k<counts b → ∀ d, xs.valueAt b h k d≠⊥)

def logicalSourceIO [Inhabited B] (M : Host.Job → Algebra B) (encode : A → B) (c : Host.Call)
    (observe : Fin c.batch → Fin c.heads → Fin c.dimensions → B → A)
    (counts : Fin c.batch → Nat) (work : List Host.Job) :
    LogicalProgramIO A B (LogicalInputs A c counts)
      (Fin c.batch → Fin c.heads → Fin c.dimensions → A) where
  core := sourceIO M encode c observe counts work
  inputs := logicalInputs A c counts
  outputs := logicalOutputs A c


def realIO (c : Host.Call) (counts : Fin c.batch → Nat) (work : List Host.Job) :
    ProgramIO (WithBot ℝ) (WithBot ℝ) :=
  sourceIO (fun _ => RealModel.model) id c (fun _ _ _ => id) counts work


end
end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.PublicIO

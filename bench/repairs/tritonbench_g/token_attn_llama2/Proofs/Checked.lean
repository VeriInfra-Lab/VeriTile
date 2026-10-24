import bench.repairs.tritonbench_g.token_attn_llama2.Host
import bench.repairs.tritonbench_g.token_attn_llama2.Integers

namespace VeriTile.Bench.Repairs.TokenAttnLlama2.Whole
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 3000000

/-- Ordinary Tensor metadata is represented separately from the checked data.
The ABI records the tested ranks, shapes, types, strides, device and view flag;
current payload binding is supplied by FlatBinding, not by an output premise. -/
structure Request (c : Call) where
  tensor : Buffer → Descriptor
  cuda : Bool
  contiguous : Buffer → Bool
  base : Buffer → Nat
  /-- Absolute bases are measured in four-byte cells. The actual pointer
  equality and alignment are ordinary Tensor ABI binding facts. -/
  address : Buffer → Nat

structure Schema (c : Call) (r : Request c) : Prop where
  shapes : (r.tensor .query).shape=[c.batches,c.heads,c.dim] ∧
    (r.tensor .key).shape=[c.cacheRows,c.kvHeads,c.dim] ∧
    (r.tensor .output).shape=[c.heads,c.outputTokens] ∧
    (r.tensor .locations).shape=[c.batches,c.maxLength] ∧
    (r.tensor .starts).shape=[c.batches] ∧ (r.tensor .lengths).shape=[c.batches]
  strides : (r.tensor .query).strides=[c.qBatch,c.qHead,c.qDim] ∧
    (r.tensor .key).strides=[c.kBatch,c.kHead,c.kDim] ∧
    (r.tensor .output).strides=[c.outHead,c.outSeq] ∧
    (r.tensor .locations).strides=[c.locBatch,c.locSeq] ∧
    (r.tensor .starts).strides=[1] ∧ (r.tensor .lengths).strides=[1]
  types : ∀ role, (r.tensor role).dtype=match role with | .query | .key | .output => .fp32 | _ => .int32
  device : ∀ role, (r.tensor role).device=(r.tensor .query).device
  ordinary : ∀ role, (r.tensor role).negative=Bool.false


structure MetadataChecks (c : Call) (m : Metadata c) : Prop where
  lengths : ∀ b : Fin c.batches, 0≤m.lengths b ∧ m.lengths b≤(c.maxLength : Int)
  starts : ∀ b : Fin c.batches, 0≤m.starts b ∧ m.begin b.val+m.len b.val≤c.outputTokens
  pages : ∀ b : Fin c.batches, ∀ t, t<m.len b.val →
    0≤finiteRead m.locations (c.locBatch*b.val+c.locSeq*(c.maxLength-m.len b.val+t)) ∧ m.page b.val t<c.cacheRows
  segments : ∀ b : Fin c.batches, ∀ b' : Fin c.batches,
    b≠b' → 0<m.len b.val → 0<m.len b'.val →
    m.begin b.val+m.len b.val≤m.begin b'.val ∨ m.begin b'.val+m.len b'.val≤m.begin b.val

/-- Checked interval disjointness implies uniqueness of every active store. -/
theorem checked_data (c : Call) (m : Metadata c) (h : MetadataChecks c m) : Data c m := by
  refine ⟨h.lengths,h.starts,h.pages,?_⟩
  intro b hb b' hb' t ht t' ht' eq
  have same : b=b' := by
    by_contra different
    have diff : (⟨b,hb⟩ : Fin c.batches)≠⟨b',hb'⟩ := by
      intro same; exact different (congrArg Fin.val same)
    have apart := h.segments ⟨b,hb⟩ ⟨b',hb'⟩ diff (by dsimp; omega) (by dsimp; omega)
    dsimp at apart
    omega
  exact ⟨same,by subst b'; omega⟩

/-- Precisely the nonempty checked branch. Backing length is an ordinary
Tensor storage property and is also compared against the pointer word limit. -/
structure Admission (c : Call) (m : Metadata c) (r : Request c) : Prop where
  schema : Schema c r
  cuda : r.cuda=Bool.true
  contiguous : ∀ role, r.contiguous role=Bool.true
  dense : Dense c
  guard : Guard c
  metadata : MetadataChecks c m
  mixedApart : ∀ f, f∈[Buffer.query,Buffer.key] → ∀ i, i∈[Buffer.locations,Buffer.starts,Buffer.lengths] →
    extent c f=0 ∨ extent c i=0 ∨ r.address f+4*extent c f≤r.address i ∨ r.address i+4*extent c i≤r.address f
  pointer : ∀ role, 0<extent c role → r.address role=4*r.base role
  backing : ∀ role, r.base role+extent c role≤2^62
  apart : ∀ role, role≠.output →
    extent c role=0 ∨ c.outSize=0 ∨
    r.base .output+c.outSize≤r.base role ∨ r.base role+extent c role≤r.base .output

/-- This decision contains only finite metadata, shape, interval and range
predicates. It neither executes a kernel nor inspects a floating payload. -/
def check (c : Call) (m : Metadata c) (r : Request c) : Bool :=
  @decide (Admission c m r) (Classical.propDecidable _)

theorem check_iff (c : Call) (m : Metadata c) (r : Request c) :
    check c m r=Bool.true ↔ Admission c m r := by simp [check]

def checkedABI (c : Call) (m : Metadata c) (r : Request c) (a : Admission c m r) : ABI c where
  tensor := r.tensor
  shapes := a.schema.shapes
  strides := a.schema.strides
  types := a.schema.types
  device := a.schema.device
  ordinary := a.schema.ordinary

def checkedPlacement (c : Call) (m : Metadata c) (r : Request c)
    (a : Admission c m r) : Placement c where
  arena := 2^62
  base := r.base
  bounded := a.backing
  outputApart := by
    intro role hr i hi o ho eq
    rcases a.apart role hr with empty | empty | left | right <;> omega

theorem checked_valid (c : Call) (m : Metadata c) (r : Request c)
    (a : Admission c m r) : Valid c m :=
  domain_valid c m a.dense a.guard (checked_data c m a.metadata)

/-- Every byte of an active physical word is in the pointer domain. -/
theorem complete_word (c : Call) (m : Metadata c) (r : Request c)
    (a : Admission c m r) (role : Buffer) (i : Nat) (hi : i<extent c role)
    (j : Nat) (hj : j<4) :
    r.address role+4*i+j<2^64 ∧
    (BitVec.ofNat 64 (r.address role+4*i+j)).toNat=r.address role+4*i+j := by
  have ptr := a.pointer role (by omega)
  have bound := a.backing role
  have step := Nat.mul_le_mul_left 4 (show r.base role+i+1≤2^62 by omega)
  have whole : r.address role+4*i+j<2^64 := by omega
  exact ⟨whole,by change _ % 2^64 = _; exact Nat.mod_eq_of_lt whole⟩

/-- Byte observation is a local encoding of the finite current physical cells.
The complete frame includes input buffers, unwritten output and arena canaries. -/
def bytes {α : Type} (encode : Cell α → Fin 4 → BitVec 8)
    (flat : Fin (2^62) → Cell α) (address : Fin (2^64)) : BitVec 8 :=
  encode (flat ⟨address.val/4,by have h:=address.isLt; omega⟩)
    ⟨address.val%4,Nat.mod_lt _ (by decide)⟩

theorem byte_frame {α : Type} [Inhabited α] (c : Call) (m : Metadata c) (r : Request c)
    (a : Admission c m r) (value : Nat → Nat → Nat → α)
    (before after : Fin (2^62) → Cell α)
    (result : FlatResult c (checkedPlacement c m r a) m value before after)
    (encode : Cell α → Fin 4 → BitVec 8) (address : Fin (2^64))
    (miss : ∀ b<c.batches, ∀ h<c.heads, ∀ t<m.len b,
      address.val/4≠r.base .output+(h*c.outHead+(m.begin b+t)*c.outSeq)) :
    bytes encode after address=bytes encode before address := by
  unfold bytes
  rw [result.2 _ miss]

/-- Empty inputs retain all header and integer-data checks, while no dense
launch layout, alignment or output/input interval check is required. -/
structure EmptySchema (c : Call) (r : Request c) : Prop where
  shapes : (r.tensor .query).shape=[c.batches,c.heads,c.dim] ∧
    (r.tensor .key).shape=[c.cacheRows,c.kvHeads,c.dim] ∧
    (r.tensor .output).shape=[c.heads,c.outputTokens] ∧
    (r.tensor .locations).shape=[c.batches,c.maxLength] ∧
    (r.tensor .starts).shape=[c.batches] ∧ (r.tensor .lengths).shape=[c.batches]
  types : ∀ role, (r.tensor role).dtype=match role with | .query | .key | .output => .fp32 | _ => .int32
  device : ∀ role, (r.tensor role).device=(r.tensor .query).device
  ordinary : ∀ role, (r.tensor role).negative=Bool.false


structure EmptyAdmission (c : Call) (m : Metadata c) (r : Request c) : Prop where
  empty : c.batches=0 ∨ c.maxLength=0
  schema : EmptySchema c r
  cuda : r.cuda=Bool.true
  contiguous : ∀ role, r.contiguous role=Bool.true
  guard : Guard c
  metadata : MetadataChecks c m

/-- Outcome of Python validation: False means a successful empty return,
True means exactly one launch, and error means no mutation. -/
def validate (c : Call) (m : Metadata c) (r : Request c) : Except Unit Bool := by
  classical
  exact if c.batches=0 ∨ c.maxLength=0 then
    if EmptyAdmission c m r then .ok Bool.false else .error ()
  else if Admission c m r then .ok Bool.true else .error ()

theorem validate_launch (c : Call) (m : Metadata c) (r : Request c) :
    validate c m r=.ok Bool.true ↔
      (0<c.batches ∧ 0<c.maxLength) ∧ Admission c m r := by
  classical
  unfold validate
  split_ifs <;> simp_all <;> omega

theorem validate_empty (c : Call) (m : Metadata c) (r : Request c) :
    validate c m r=.ok Bool.false ↔ EmptyAdmission c m r := by
  classical
  unfold validate
  split_ifs <;> simp_all
  all_goals
    intro h
    have e := h.empty
    omega

/-- Public wrapper dispatch. Empty validation returns the current state and
exact supplied descriptor; the dense ABI is constructed only for a launch. -/
def dispatch {α : Type} [Inhabited α] (A : Program → Algebra α) (c : Call)
    (m : Metadata c) (r : Request c) (scale : ℝ) (work : List Program) (s : State α) :
    Except Unit (Option (Returned α)) := by
  classical
  exact if he : validate c m r=.ok Bool.false then .ok (some ⟨s,r.tensor .output,none⟩)
    else if hl : validate c m r=.ok Bool.true then
      .ok (invoke A c scale (checkedABI c m r ((validate_launch c m r).mp hl).2) work s)
    else .error ()

theorem dispatch_empty {α : Type} [Inhabited α] (A : Program → Algebra α) (c : Call)
    (m : Metadata c) (r : Request c) (scale : ℝ) (work : List Program) (s : State α)
    (h : validate c m r=.ok Bool.false) :
    dispatch A c m r scale work s=.ok (some ⟨s,r.tensor .output,none⟩) := by simp [dispatch,h]

theorem dispatch_launch {α : Type} [Inhabited α] (A : Program → Algebra α) (c : Call)
    (m : Metadata c) (r : Request c) (scale : ℝ) (work : List Program) (s : State α)
    (h : validate c m r=.ok Bool.true) :
    dispatch A c m r scale work s=.ok
      (invoke A c scale (checkedABI c m r ((validate_launch c m r).mp h).2) work s) := by
  simp [dispatch,h]

/-- The empty wrapper branch returns None and preserves the actual caller
memory without invoking a kernel, even for Torch's noncanonical empty strides. -/
def emptyCall {α : Type} (output : Descriptor) (memory : α) := ((none : Option Unit),output,memory)

theorem empty_call {α : Type} (output : Descriptor) (memory : α) :
    (emptyCall output memory).1=none ∧ (emptyCall output memory).2.1=output ∧
      (emptyCall output memory).2.2=memory := ⟨rfl,rfl,rfl⟩

#axiomsClean checked_data
#axiomsClean checked_valid
#axiomsClean complete_word
#axiomsClean byte_frame
#specNonCircular Admission avoiding [FP.Structural.exec,invoke,NativeGrid.runPrograms]
end
end VeriTile.Bench.Repairs.TokenAttnLlama2.Whole

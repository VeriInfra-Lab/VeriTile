import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Tensor
import bench.repairs.tritonbench_g.token_attn_llama2_grid_axes_checked.Proofs.Integers

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole
open VeriTile.Triton FP.Structural
noncomputable section
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


end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked.Whole

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton FP.Structural Whole
noncomputable section
open Classical
set_option maxHeartbeats 4000000

def active (t : TensorCall) : Prop := 0 < t.call.batches ∧ 0 < t.maximum

structure Header (t : TensorCall) : Prop where
  q : t.q.tensor.metadata.shape=[t.call.batches,t.call.heads,t.call.dim]
  k : t.k.tensor.metadata.shape=[t.call.cacheRows,t.call.kvHeads,t.call.dim]
  output : t.output.tensor.metadata.shape=[t.call.heads,t.call.outputTokens]
  locations : t.locations.metadata.shape=[t.call.batches,t.maximum]
  starts : t.starts.metadata.shape=[t.call.batches]
  lengths : t.lengths.metadata.shape=[t.call.batches]
  floatTypes : ∀ p∈[t.q,t.k,t.output],p.tensor.metadata.dtype=.floating .fp32
  integerTypes : ∀ p∈[t.locations,t.starts,t.lengths],p.metadata.dtype=.other "int32"
  cuda : t.q.tensor.metadata.device.kind="cuda"
  devices : ∀ o∈t.observations,o.metadata.device=t.q.tensor.metadata.device
  contiguous : ∀ o∈t.observations,o.metadata.isContiguous=Bool.true
  ordinary : ∀ o∈t.observations,o.metadata.isNeg=Bool.false

/-- All six exact source stride tuples, conditional only on an actual launch. -/
def DenseChecks (t : TensorCall) : Prop :=
  t.q.strides=[t.call.heads*t.call.dim,t.call.dim,1] ∧
  t.k.strides=[t.call.kvHeads*t.call.dim,t.call.dim,1] ∧
  t.output.strides=[t.call.outputTokens,1] ∧
  t.locations.strides=[t.maximum,1] ∧ t.starts.strides=[1] ∧ t.lengths.strides=[1]

def Apart (a b : Observation) : Prop :=
  a.metadata.numel=0 ∨ b.metadata.numel=0 ∨
  a.metadata.dataPtr.toNat+4*a.metadata.numel≤b.metadata.dataPtr.toNat ∨
  b.metadata.dataPtr.toNat+4*b.metadata.numel≤a.metadata.dataPtr.toNat

def PhysicalChecks (t : TensorCall) : Prop :=
  (∀ f∈[t.q.observation,t.k.observation],
    ∀ i∈[t.locations.observation,t.starts.observation,t.lengths.observation],Apart f i) ∧
  (∀ o∈t.observations,o.metadata.dataPtr.toNat+4*o.metadata.numel≤2^64 ∧
    (0 < o.metadata.numel → o.metadata.dataPtr.toNat%4=0)) ∧
  (∀ o∈[t.q.observation,t.k.observation,t.locations.observation,t.starts.observation,t.lengths.observation],
    Apart t.output.observation o)

/-- The complete and only added domain restriction. -/
def AxisChecks (t : TensorCall) : Prop :=
  t.call.heads≤65535 ∧ (t.maximum+31)/32≤65535

/-- The inherited quotient-form head facts are precisely the Python positive
divisible-head test; they add no hidden restriction. -/
theorem head_checks_iff (heads keys : Nat) :
    (0 < keys ∧ 0 < heads ∧ heads%keys=0) ↔
      (0 < keys ∧ 0 < heads/keys ∧ heads=keys*(heads/keys)) := by
  constructor
  · rintro ⟨hk,hh,hm⟩
    have eq : heads=keys*(heads/keys) := by
      have division := Nat.mod_add_div heads keys
      simpa [hm] using division.symm
    refine ⟨hk,?_,eq⟩
    apply Nat.pos_of_ne_zero
    intro zero
    rw [zero,Nat.mul_zero] at eq
    omega
  · rintro ⟨hk,hq,eq⟩
    have positive := Nat.mul_pos hk hq
    rw [←eq] at positive
    refine ⟨hk,positive,?_⟩
    conv_lhs => rw [eq]
    simp

def InputContract (t : TensorCall) (words : Nat → Word) : Prop :=
  Header t ∧ Guard t.call ∧ MetadataChecks t.call (metadata t words) ∧
    (active t → DenseChecks t ∧ PhysicalChecks t ∧ AxisChecks t)

/-- Successful Python guard outcomes. Header/integer classification precedes
this Tensor-and-Nat observation; no payload value or result is inspected. -/
def checkInput (t : TensorCall) (words : Nat → Word) : Except String Bool :=
  if ¬Header t then .error "Tensor header"
  else if ¬Guard t.call then .error "shape or integer envelope"
  else if ¬MetadataChecks t.call (metadata t words) then .error "integer contents"
  else if active t then
    if DenseChecks t ∧ PhysicalChecks t ∧ AxisChecks t then .ok Bool.true
    else .error "launch metadata"
  else .ok Bool.false

theorem checks_iff (t : TensorCall) (words : Nat → Word) :
    (∃ launch, checkInput t words=.ok launch) ↔ InputContract t words := by
  unfold checkInput InputContract
  split_ifs <;> simp_all

theorem launch_checks_iff (t : TensorCall) (words : Nat → Word) :
    checkInput t words=.ok Bool.true ↔ active t ∧ InputContract t words := by
  unfold checkInput InputContract
  split_ifs <;> simp_all

theorem empty_checks_iff (t : TensorCall) (words : Nat → Word) :
    checkInput t words=.ok Bool.false ↔ ¬active t ∧ InputContract t words := by
  unfold checkInput InputContract
  split_ifs <;> simp_all

theorem dense (t : TensorCall) (header : Header t) (h : DenseChecks t) : Dense t.call := by
  refine ⟨?_,?_,?_,?_,?_⟩
  · change t.q.strides[0]?.getD 0=t.call.heads*t.call.dim ∧
      t.q.strides[1]?.getD 0=t.call.dim ∧ t.q.strides[2]?.getD 0=1
    rw [h.1]
    simp only [List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,and_self]
  · change t.k.strides[0]?.getD 0=t.call.kvHeads*t.call.dim ∧
      t.k.strides[1]?.getD 0=t.call.dim ∧ t.k.strides[2]?.getD 0=1
    rw [h.2.1]
    simp only [List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,and_self]
  · change t.locations.strides[0]?.getD 0=t.maximum ∧ t.locations.strides[1]?.getD 0=1
    rw [h.2.2.2.1]
    simp only [List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,and_self]
  · change t.output.strides[0]?.getD 0=t.call.outputTokens ∧ t.output.strides[1]?.getD 0=1
    rw [h.2.2.1]
    simp only [List.getElem?_cons_zero,List.getElem?_cons_succ,Option.getD_some,and_self]
  · change t.q.tensor.metadata.numel=_ ∧ t.k.tensor.metadata.numel=_ ∧
      t.locations.metadata.numel=_ ∧ t.output.tensor.metadata.numel=_
    unfold VeriTile.Torch.TensorMetadata.numel
    rw [header.q,header.k,header.locations,header.output]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one,Nat.mul_assoc,
      show t.call.maxLength=t.maximum from rfl,and_self]

theorem valid (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) : Valid t.call (metadata t words) :=
  domain_valid t.call (metadata t words) (dense t contract.1 (contract.2.2.2 launch).1)
    contract.2.1 (checked_data t.call (metadata t words) contract.2.2.1)

theorem launch_axes (t : TensorCall) (words : Nat → Word) (contract : InputContract t words)
    (launch : active t) : t.call.batches < 2^31 ∧ t.call.heads≤65535 ∧
      (t.maximum+31)/32≤65535 := by
  have bound := contract.2.1.signed
  unfold envelope at bound
  exact ⟨by omega,(contract.2.2.2 launch).2.2⟩

theorem empty_domain_unchanged (t : TensorCall) (words : Nat → Word) (empty : ¬active t) :
    InputContract t words ↔ Header t ∧ Guard t.call ∧ MetadataChecks t.call (metadata t words) := by
  simp [InputContract,empty]

#axiomsClean checks_iff
#axiomsClean head_checks_iff
#axiomsClean launch_checks_iff
#axiomsClean empty_checks_iff
#axiomsClean dense
#axiomsClean valid
#axiomsClean launch_axes
#axiomsClean empty_domain_unchanged
end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked

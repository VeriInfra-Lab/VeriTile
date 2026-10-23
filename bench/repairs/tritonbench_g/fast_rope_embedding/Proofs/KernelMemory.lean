import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.KernelSource

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section
open scoped Classical
set_option maxHeartbeats 4000000

/-- One physical reader supplies Q and both readonly tables, including aliases. -/
def state {α β : Type} [Inhabited α] (embed : β→α) (read : Nat→β) (inputs : Inputs) : State α where
  mem := fun region offset=>.mk .real (embed (read (
    (if region="Q" then inputs.q.dataPtr.toNat else
      if region="C" then inputs.cosine.dataPtr.toNat else inputs.sine.dataPtr.toNat)+4*offset)))
  regs := fun _ _ _=>none
  pids := fun _=>0
  numPids := fun _=>1
  undef := fun d _ _=>defaultValue d

def logicalRead (read : Nat→β) (tensor : VeriTile.Torch.FP32.Tensor) (index : Nat) : β :=
  read (tensor.storage.address (tensor.physicalIndex index))

/-- Actual Tensor coordinates, derived from ordinary contiguous Tensor laws. -/
theorem tensor_address (t : VeriTile.Torch.FP32.Tensor) (laws : VeriTile.Torch.FP32.TensorLaws t)
    (payload : payloadCheck t.metadata=Bool.true) (k : Nat) (hk : k<t.metadata.numel) :
    t.storage.address (t.physicalIndex k)=t.dataPtr.toNat+4*k := by
  have hc := (KernelBinding.payload_facts _ payload).1
  have index := laws.contiguous hc k hk
  have pointer := congrArg BitVec.toNat (laws.pointer (by omega))
  have cap := laws.inBounds k hk
  have span := t.storage.spanFits
  rw [index] at cap ⊢
  have bound : t.storage.address t.storageOffset<2^64 := by
    simp only [VeriTile.Torch.FP32.Storage.address] at *
    omega
  simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt bound] at pointer
  rw [pointer]
  simp [VeriTile.Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

/-- Real observations use an explicit interpretation of FP32 words. Native
observations retain the same words because lazy negative views were rejected. -/
theorem logical_word (read : Nat→Word) (t : VeriTile.Torch.FP32.Tensor)
    (laws : VeriTile.Torch.FP32.TensorLaws t) (payload : payloadCheck t.metadata=Bool.true) (k : Nat) :
    logicalRead read t k=VeriTile.Torch.FP32.logicalValue read t laws k := by
  simp [logicalRead,VeriTile.Torch.FP32.logicalValue,(KernelBinding.payload_facts _ payload).2]

/-- Physical realization of the completed Q region. Outside the actual source
writes the source frame proves equality to the entry reader. -/
def physical {α β : Type} [Inhabited α] (project : α→β) (inputs : Inputs)
    (before : Nat→β) (final : State α) (address : Nat) : β :=
  if inputs.q.dataPtr.toNat≤address ∧ (address-inputs.q.dataPtr.toNat)%4=0 then
    project ((final.mem "Q" ((address-inputs.q.dataPtr.toNat)/4)).read .real)
  else before address

theorem physical_read {α β : Type} [Inhabited α] (project : α→β) (inputs : Inputs)
    (before : Nat→β) (final : State α) (offset : Nat) :
    physical project inputs before final (inputs.q.dataPtr.toNat+4*offset)=
      project ((final.mem "Q" offset).read .real) := by
  simp [physical]

/-- Complete physical word footprint of this source program, at byte addresses. -/
def WrittenAddress (c : Call) (row group address : Nat) : Prop :=
  ∃ offset,Kernel.Written c.geometry row group "Q" offset ∧
    address=c.inputs.q.dataPtr.toNat+4*offset

/-- The physical interpretation commutes with an individual FP32 source store.
No output equation or completed kernel result is an interpretation premise. -/
theorem store_physical {α β : Type} [Inhabited α] (project : α→β) (inputs : Inputs)
    (before : Nat→β) (s : State α) (offset : Nat) (value : α) (address : Nat) :
    physical project inputs before (s.write "Q" offset (.mk .real value)) address=
      if address=inputs.q.dataPtr.toNat+4*offset then project value
      else physical project inputs before s address := by
  unfold physical
  by_cases h : inputs.q.dataPtr.toNat≤address ∧ (address-inputs.q.dataPtr.toNat)%4=0
  · simp only [if_pos h]
    have eq : (address-inputs.q.dataPtr.toNat)/4=offset ↔ address=inputs.q.dataPtr.toNat+4*offset := by omega
    by_cases same : (address-inputs.q.dataPtr.toNat)/4=offset
    · rw [if_pos (eq.mp same)]
      simp [State.write,same]
    · rw [if_neg (fun hit=>same (eq.mpr hit))]
      simp [State.write,same]
  · have neq : address≠inputs.q.dataPtr.toNat+4*offset := by intro he; simp [he] at h
    simp [h,neq]

theorem physical_frame {α β : Type} [Inhabited α] (embed : β→α) (project : α→β)
    (retract : ∀ v,project (embed v)=v) (c : Call) (row group : Nat)
    (before : Nat→β) (final : State α) (value : Nat→Nat→α)
    (post : Kernel.Post c.geometry row group (state embed before c.inputs) final value)
    (address : Nat) (outside : ¬WrittenAddress c row group address) :
    physical project c.inputs before final address=before address := by
  unfold physical
  split
  · rename_i hit
    have eq : c.inputs.q.dataPtr.toNat+4*((address-c.inputs.q.dataPtr.toNat)/4)=address := by omega
    have miss : ¬Kernel.Written c.geometry row group "Q" ((address-c.inputs.q.dataPtr.toNat)/4) := by
      intro hw
      exact outside ⟨_,hw,eq.symm⟩
    rw [post.frame "Q" _ miss]
    simpa [state,eq] using retract (before address)
  · rfl

/-- The written set exactly characterizes which logical coordinates change. -/
theorem written_coordinate (c : Call) (row group peer head col : Nat)
    (hh : head<c.geometry.heads) (hc : col<c.geometry.dimension) :
    Kernel.Written c.geometry row group "Q" (c.geometry.offset peer head col) ↔
      peer=row ∧ group*4≤head ∧ head<min (group*4+4) c.geometry.heads ∧ col<2*c.geometry.half := by
  constructor
  · rintro ⟨_,h,j,hlo,hhi,hj,eq⟩
    have dim : 2*c.geometry.half≤c.geometry.dimension := by unfold Request.half; omega
    have inj := offset_injective c.geometry hh hc (lt_of_lt_of_le hhi (min_le_right _ _)) (by omega) eq
    rcases inj with ⟨rfl,rfl,rfl⟩
    exact ⟨rfl,hlo,hhi,hj⟩
  · rintro ⟨rfl,hlo,hhi,hj⟩
    exact ⟨rfl,head,col,hlo,hhi,hj,rfl⟩

def affected (c : Call) (row group peer head col : Nat) : Prop :=
  peer=row ∧ group*4≤head ∧ head<min (group*4+4) c.geometry.heads ∧ col<2*c.geometry.half

/-- Every logical Q output is read from the same physical final memory. This
includes other groups/rows, odd tails, and all untouched Tensor coordinates. -/
theorem physical_tensor {α β : Type} [Inhabited α] (embed : β→α) (project : α→β)
    (retract : ∀ v,project (embed v)=v) (c : Call) (admit : Admitted c) (row group : Nat)
    (before : Nat→β) (final : State α) (value : Nat→Nat→α)
    (post : Kernel.Post c.geometry row group (state embed before c.inputs) final value)
    (peer head col : Nat) (hr : peer<c.geometry.rows) (hh : head<c.geometry.heads) (hc : col<c.geometry.dimension) :
    logicalRead (physical project c.inputs before final) c.inputs.q (c.geometry.offset peer head col)=
      if affected c row group peer head col then project (value head col)
      else logicalRead before c.inputs.q (c.geometry.offset peer head col) := by
  classical
  have bound : c.geometry.offset peer head col<c.inputs.q.metadata.numel := by
    rw [q_size c admit]
    exact offset_lt c.geometry peer head col hr hh hc
  have addr := tensor_address c.inputs.q c.inputs.qLaws
    (KernelBinding.admitted_payload c.inputs c.geometry admit.metadata).1 _ bound
  simp only [logicalRead,addr,physical_read]
  split
  · rename_i active
    rcases active with ⟨rfl,hlo,hhi,hcol⟩
    rw [post.values head col hlo hhi hcol,Cell.read_same]
  · rename_i miss
    have outside := (written_coordinate c row group peer head col hh hc).not.mpr miss
    rw [post.frame "Q" _ outside]
    simpa [state] using retract (before (c.inputs.q.dataPtr.toNat+4*c.geometry.offset peer head col))

#axiomsClean store_physical
#axiomsClean physical_tensor
#axiomsClean physical_frame
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.KernelContract

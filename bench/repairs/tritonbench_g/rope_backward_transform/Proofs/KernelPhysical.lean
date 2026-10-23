import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.KernelResult

namespace VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract
open VeriTile.Triton FP.Structural
noncomputable section
open scoped Classical
set_option maxHeartbeats 5000000

def WrittenByte (c : Call) (key : Bool) (address : Nat) : Prop :=
  (c.q key).dataPtr.toNat≤address ∧ Written c.geometry key ((address-(c.q key).dataPtr.toNat)/4)

def WrittenWord (c : Call) (key : Bool) (address : Nat) : Prop :=
  WrittenByte c key address ∧ (address-(c.q key).dataPtr.toNat)%4=0

def InStorage (t : Tensor) (address : Nat) : Prop :=
  t.storage.base≤address ∧ address<t.storage.base+4*t.storage.elements

theorem apart_points (a b : Tensor) (disjoint : apart a b) (address : Nat)
    (ha : InStorage a address) (hb : InStorage b address) : False := by
  rcases disjoint.2 with h | h | h | h <;> unfold InStorage at * <;> omega

theorem coordinate_span (c : Call) (admit : Admitted c) (key : Bool) (offset : Nat)
    (bound : offset<c.geometry.size key) (byte : Fin 4) :
    InStorage (c.q key) ((c.q key).dataPtr.toNat+4*offset+byte.val) := by
  have h : offset<(c.q key).metadata.numel := by rw [q_numel c admit key]; exact bound
  have cap := (c.qLaws key).inBounds offset h
  have address := q_address c admit key offset bound
  unfold InStorage VeriTile.Torch.FP32.Storage.address at *
  omega

theorem written_span (c : Call) (admit : Admitted c) (key : Bool) (address : Nat)
    (written : WrittenByte c key address) : InStorage (c.q key) address := by
  have offset := written_bound c.geometry key _ written.2
  let byte : Fin 4 := ⟨(address-(c.q key).dataPtr.toNat)%4,Nat.mod_lt _ (by decide)⟩
  have span := coordinate_span c admit key _ offset byte
  have eq : (c.q key).dataPtr.toNat+4*((address-(c.q key).dataPtr.toNat)/4)+byte.val=address := by
    dsimp [byte]; have lower:=written.1; omega
  rwa [eq] at span

theorem opposite_apart (c : Call) (admit : Admitted c) (key : Bool) (address : Nat)
    (inside : InStorage (c.q key) address) : ¬WrittenByte c (!key) address := by
  intro written
  have other:=written_span c admit (!key) address written
  cases key
  · exact apart_points _ _ admit.qApart address inside other
  · exact apart_points _ _ admit.qApart address other inside

theorem written_apart (c : Call) (admit : Admitted c) (address : Nat) :
    ¬(WrittenByte c Bool.false address ∧ WrittenByte c Bool.true address) := by
  rintro ⟨q,k⟩
  exact apart_points _ _ admit.qApart address (written_span c admit Bool.false address q)
    (written_span c admit Bool.true address k)

theorem written_coordinate (c : Call) (key : Bool) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key) (hc : col<c.geometry.dimension) :
    Written c.geometry key (c.geometry.offset key row head col) ↔ col<2*c.geometry.half := by
  constructor
  · rintro ⟨peer,h,j,_,hhead,hcol,equal⟩
    have dim : 2*c.geometry.half≤c.geometry.dimension := by unfold Request.half; omega
    have same:=offset_injective c.geometry key hh hc hhead (by omega) equal
    omega
  · intro active
    exact ⟨row,head,col,hr,hh,active,rfl⟩

def physical {α β : Type} [Inhabited α] (project : α→β) (c : Call)
    (before : Nat→β) (final : Source.Memory α) (address : Nat) : β :=
  if WrittenWord c Bool.false address then
    project ((final "Q" ((address-(c.q Bool.false).dataPtr.toNat)/4)).read .fp32)
  else if WrittenWord c Bool.true address then
    project ((final "K" ((address-(c.q Bool.true).dataPtr.toNat)/4)).read .fp32)
  else before address

theorem physical_coordinate {α β : Type} [Inhabited α] (project : α→β) (c : Call)
    (admit : Admitted c) (before : Nat→β) (final : Source.Memory α) (key : Bool)
    (offset : Nat) (bound : offset<c.geometry.size key) :
    physical project c before final ((c.q key).dataPtr.toNat+4*offset)=
      if Written c.geometry key offset then project ((final (region key) offset).read .fp32)
      else before ((c.q key).dataPtr.toNat+4*offset) := by
  have span := coordinate_span c admit key offset bound (⟨0,by decide⟩ : Fin 4)
  simp only [Nat.add_zero] at span
  have other := opposite_apart c admit key _ span
  have own : WrittenByte c key ((c.q key).dataPtr.toNat+4*offset) ↔ Written c.geometry key offset := by
    simp [WrittenByte]
  cases key
  · simp only [Bool.not_false] at other
    simp [physical,own,WrittenWord,other,region]
  · simp only [Bool.not_true] at other
    simp [physical,own,WrittenWord,other,region]

theorem physical_frame {α β : Type} [Inhabited α] (project : α→β) (c : Call)
    (before : Nat→β) (final : Source.Memory α) (address : Nat)
    (outside : ¬WrittenWord c Bool.false address ∧ ¬WrittenWord c Bool.true address) :
    physical project c before final address=before address := by simp [physical,outside.1,outside.2]

theorem physical_values {α β : Type} [Inhabited α] (op : Operations α)
    (embed : β→α) (project : α→β) (retract : ∀x,project (embed x)=x)
    (c : Call) (admit : Admitted c) (before : Nat→β) (key : Bool) (row head col : Nat)
    (hr : row<c.geometry.rows) (hh : head<c.geometry.heads key) (hc : col<c.geometry.dimension) :
    logicalRead (physical project c before
      (grid op.backend c.geometry (memory embed before c) (launchRows c.geometry)))
      (c.q key) (c.geometry.offset key row head col)=
      project (tensorScalar op embed c before key row head col) := by
  have bound:=offset_bound c.geometry key row head col hr hh hc
  rw [logicalRead,q_address c admit key _ bound,physical_coordinate project c admit before _ key _ bound]
  rw [written_coordinate c key row head col hr hh hc]
  split
  · rw [grid_values op embed c admit before key row head col hr hh hc]
  · rename_i inactive
    simp only [tensorScalar,if_neg inactive,retract,logicalRead,q_address c admit key _ bound]

#axiomsClean coordinate_span
#axiomsClean written_apart
#axiomsClean physical_values
#axiomsClean physical_frame
end
end VeriTile.Bench.Repairs.RopeBackwardTransform.KernelContract

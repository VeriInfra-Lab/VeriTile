import bench.repairs.tritonbench_g.chunk_retention.Proofs.BackwardBinding

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked
noncomputable section
open HostABI
set_option maxHeartbeats 3500000

/-- The source forms padded block-pointer lanes even when their load/store mask
is false. These envelopes use source dimensions rather than logical capacity. -/
theorem padded_key_extent (c : Shape) (h r k : Nat)
    (hh : h<c.heads) (hr : r<c.paddedTime) (hk : k<c.paddedKey) :
    c.keyOffset h r k<keyExtent c := by
  have hc : 0<c.chunks := by simp only [Shape.paddedTime] at hr;omega
  have h1:=Nat.mul_le_mul_right c.K (Nat.mul_le_mul_right c.T (Nat.le_sub_one_of_lt hh))
  have h2:=Nat.mul_le_mul_right c.K (Nat.le_sub_one_of_lt hr)
  simp only [keyExtent,if_pos (show 0<c.heads ∧ 0<c.chunks from ⟨by omega,hc⟩),Shape.keyOffset,Shape.keyHead,←Nat.mul_assoc]
  omega

theorem padded_value_extent (c : Shape) (h r v : Nat)
    (hh : h<c.heads) (hr : r<c.paddedTime) (hv : v<c.paddedValue) :
    c.valueOffset h r v<valueExtent c := by
  have hc : 0<c.chunks := by simp only [Shape.paddedTime] at hr;omega
  have h1:=Nat.mul_le_mul_right c.V (Nat.mul_le_mul_right c.T (Nat.le_sub_one_of_lt hh))
  have h2:=Nat.mul_le_mul_right c.V (Nat.le_sub_one_of_lt hr)
  simp only [valueExtent,if_pos (show 0<c.heads ∧ 0<c.chunks from ⟨by omega,hc⟩),Shape.valueOffset,Shape.valueHead,←Nat.mul_assoc]
  omega

theorem padded_state_extent (c : Shape) (h k v : Nat)
    (hh : h<c.heads) (hk : k<c.paddedKey) (hv : v<c.paddedValue) :
    c.stateOffset h k v<stateExtent c := by
  have h1:=Nat.mul_le_mul_right c.V (Nat.mul_le_mul_right c.K (Nat.le_sub_one_of_lt hh))
  have h2:=Nat.mul_le_mul_right c.V (Nat.le_sub_one_of_lt hk)
  simp only [stateExtent,if_pos (by omega : 0<c.heads),Shape.stateOffset,Shape.stateHead,←Nat.mul_assoc]
  omega

theorem padded_history_extent (c : Shape) (h t k v : Nat)
    (hh : h<c.heads) (ht : t<c.chunks) (hk : k<c.paddedKey) (hv : v<c.paddedValue) :
    c.historyOffset h t k v<historyExtent c := by
  have h1:=Nat.mul_le_mul_right c.V (Nat.mul_le_mul_right c.K
    (Nat.mul_le_mul_right c.chunks (Nat.le_sub_one_of_lt hh)))
  have h2:=Nat.mul_le_mul_right c.V (Nat.mul_le_mul_right c.K (Nat.le_sub_one_of_lt ht))
  have h3:=Nat.mul_le_mul_right c.V (Nat.le_sub_one_of_lt hk)
  simp only [historyExtent,if_pos (by omega : 0<c.heads ∧ 0<c.chunks),
    Shape.historyOffset,Shape.historyHead,Shape.stateHead,←Nat.mul_assoc]
  omega

theorem padded_partial_extent (c : Shape) (b h r v : Nat)
    (hb : b<c.keyBlocks) (hh : h<c.heads) (hr : r<c.paddedTime) (hv : v<c.paddedValue) :
    c.partialOffset b h r v<partialExtent c := by
  have hc : 0<c.chunks := by simp only [Shape.paddedTime] at hr;omega
  have bh:=digit_bound b c.keyBlocks h c.heads hb hh
  have h1:=Nat.mul_le_mul_right c.V (Nat.mul_le_mul_right c.T (Nat.le_sub_one_of_lt bh))
  have h2:=Nat.mul_le_mul_right c.V (Nat.le_sub_one_of_lt hr)
  simp only [partialExtent,if_pos (show 0<c.heads ∧ 0<c.chunks from ⟨by omega,hc⟩),Shape.partialOffset,Shape.valueOffset,
    Shape.valueSize,Shape.valueHead,←Nat.mul_assoc]
  simp only [Nat.add_mul] at h1
  omega

/-- All active FP32 bytes are storage-backed; every formed padded address is
naturally aligned and has representable signed displacement/unsigned address. -/
def PointerSafety (size : Nat) (t : Tensor) : Prop :=
  (∀ n<t.numel,∀ j<4,t.storageBase≤t.dataPointer+4*n+j ∧
    t.dataPointer+4*n+j<t.storageBase+t.storageBytes ∧ t.dataPointer+4*n+j<2^64) ∧
  (∀ n<size,∀ j<4,4*n+j<2^63 ∧ t.dataPointer%4=0 ∧ t.dataPointer+4*n+j<2^64)

theorem pointer_safety (size : Nat) (t : Tensor) (checked : pointerChecks size t=Bool.true)
    (signed : 4*size<2^63) : PointerSafety size t :=
  ⟨active_bytes size t checked,formed_bytes size t checked signed⟩

theorem forward_extent_inventory (c : Shape) (f : ForwardObjects c) (size : Nat) (t : Tensor)
    (member : (size,t)∈forwardPointers c f) : size∈extents c := by
  simp only [forwardPointers,List.mem_append,List.mem_cons,List.not_mem_nil,or_false,List.mem_map] at member
  rcases member with ((h|h|h|h|h)|h)|h
  all_goals first
    | (have eq:=congrArg Prod.fst h;simp only [Prod.fst] at eq;subst size;simp [extents])
    | (obtain ⟨u,hu,he⟩:=h;have eq:=congrArg Prod.fst he;simp only [Prod.fst] at eq;subst size;simp [extents])

theorem backward_extent_inventory (c : Shape) (f : ForwardObjects c) (b : BackwardObjects c f)
    (size : Nat) (t : Tensor) (member : (size,t)∈backwardPointers c f b) : size∈extents c := by
  simp only [backwardPointers,List.mem_append,List.mem_cons,List.not_mem_nil,or_false,List.mem_map] at member
  rcases member with (h|h|h|h|h|h|h|h|h|h)|h
  all_goals first
    | (have eq:=congrArg Prod.fst h;simp only [Prod.fst] at eq;subst size;simp [extents])
    | (obtain ⟨u,hu,he⟩:=h;have eq:=congrArg Prod.fst he;simp only [Prod.fst] at eq;subst size;simp [extents])

/-- No numeric law, Real bridge, or future backward observation is needed. -/
theorem forward_structural_safety (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (checks : forwardChecks (checkedShape g hs) f=Bool.true) :
    Accepted (checkedShape g hs) ∧
    (0<(checkedShape g hs).heads → (gridH (checkedShape g hs)).Legal ∧
      (gridO (checkedShape g hs)).Legal ∧ (gridQKV (checkedShape g hs)).Legal) ∧
    (∀ size t,(size,t)∈forwardPointers (checkedShape g hs) f → PointerSafety size t) := by
  have hc:=checks
  simp only [forwardChecks,Bool.and_eq_true,List.all_eq_true] at hc
  have ok:=plan_admitted g hs hc.1.1
  refine ⟨ok,launch_axes _ ok,?_⟩
  intro size t member
  exact pointer_safety size t (hc.2 (size,t) member)
    (ok.extents64 size (forward_extent_inventory _ f size t member))

theorem backward_structural_safety (g : Geometry) (hs : shapeChecks g=Bool.true)
    (f : ForwardObjects (checkedShape g hs)) (b : BackwardObjects (checkedShape g hs) f)
    (checks : backwardChecks (checkedShape g hs) f b=Bool.true) :
    Accepted (checkedShape g hs) ∧
    (0<(checkedShape g hs).heads → (gridH (checkedShape g hs)).Legal ∧
      (gridO (checkedShape g hs)).Legal ∧ (gridQKV (checkedShape g hs)).Legal) ∧
    (∀ size t,(size,t)∈backwardPointers (checkedShape g hs) f b → PointerSafety size t) := by
  have hc:=checks
  simp only [backwardChecks,Bool.and_eq_true,List.all_eq_true] at hc
  have ok:=plan_admitted g hs hc.1.1.1
  refine ⟨ok,launch_axes _ ok,?_⟩
  intro size t member
  exact pointer_safety size t (hc.2 (size,t) member)
    (ok.extents64 size (backward_extent_inventory _ f b size t member))

#axiomsClean padded_key_extent
#axiomsClean padded_value_extent
#axiomsClean padded_state_extent
#axiomsClean padded_history_extent
#axiomsClean padded_partial_extent
#axiomsClean forward_structural_safety
#axiomsClean backward_structural_safety
end
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.Checked

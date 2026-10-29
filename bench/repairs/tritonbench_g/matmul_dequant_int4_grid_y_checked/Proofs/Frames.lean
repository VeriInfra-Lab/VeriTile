import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Tensor

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Frames
open MatmulDequantInt4.Direct Tensor
noncomputable section

private theorem positive (p : PackedInfo) (f : Source.Profile) (o : Info) (h : OutputABI p f o) :
    0<o.tensor.storageBytes := by
  have ordinary := h.1.2.2.2.2.2.2.1
  have capacity := ordinary.2.2.2.2.2.2.2.2.2.2.2.1
  have width : 0<f.hostDtype.width := by cases f <;> decide
  exact lt_of_lt_of_le (Nat.mul_pos (Nat.mul_pos width ordinary.2.2.1) ordinary.2.2.2.1) capacity

/-- Equal storage identities denote one allocation. The already checked
interval separation therefore entails distinct input/output allocation names. -/
theorem allocations (p : PackedInfo) (f : Source.Profile) (o : Info) (t : Arguments p o)
    (h : OutputABI p f o) :
    t.weight.storageObject≠t.output.storageObject ∧
    t.scale.storageObject≠t.output.storageObject ∧
    t.zero.storageObject≠t.output.storageObject := by
  have pos := positive p f o h
  have separate (info : MatmulDequantInt4.Host.Tensor) (member : info∈[p.weight.tensor,p.scale.tensor,p.zero.tensor])
      (s : Nat) (role : (info=p.weight.tensor ∧ s=t.weight.storageObject) ∨
        (info=p.scale.tensor ∧ s=t.scale.storageObject) ∨ (info=p.zero.tensor ∧ s=t.zero.storageObject)) :
      s≠t.output.storageObject := by
    intro same
    obtain ⟨base,bytes⟩ := t.output_storage info member s role same
    have apart := h.2.2 info member
    unfold MatmulDequantInt4.Host.Disjoint at apart
    rw [base,bytes] at apart
    omega
  exact ⟨separate _ (by simp) _ (Or.inl ⟨rfl,rfl⟩),
    separate _ (by simp) _ (Or.inr (Or.inl ⟨rfl,rfl⟩)),
    separate _ (by simp) _ (Or.inr (Or.inr ⟨rfl,rfl⟩))⟩

/-- The full Tensor postcondition entails preservation of each named input
allocation, including capacity beyond the active coordinates. -/
theorem inputs {α : Type} (p : PackedInfo) (f : Source.Profile) (o : Info) (t : Arguments p o)
    (h : OutputABI p f o) (target : Nat → Nat → α) (initial final : Heap α)
    (post : Post p o t target initial final) :
    (∀ storage∈[t.weight.storageObject,t.scale.storageObject,t.zero.storageObject],
      ∀ a,final.values storage a=initial.values storage a) ∧ final.packed=initial.packed := by
  have names := allocations p f o t h
  refine ⟨?_,post.2.2⟩
  intro storage member a
  have ne : storage≠t.output.storageObject := by
    simp only [List.mem_cons,List.not_mem_nil,or_false] at member
    rcases member with rfl|rfl|rfl
    · exact names.1
    · exact names.2.1
    · exact names.2.2
  exact post.2.1 storage a (Or.inl ne)

#axiomsClean allocations
#axiomsClean inputs
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.Frames

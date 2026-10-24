import bench.repairs.tritonbench_g.layer_norm_liger.Contract
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.ForwardResult
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.Frame

namespace VeriTile.Bench.Repairs.LayerNormLiger.KernelContract
open VeriTile.Triton FP.Structural
open VeriTile.Torch
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSectionVars false
set_option linter.unusedVariables false
variable {α : Type} [Inhabited α]
variable {b : Buffer} {shape : List Nat} {pointers : RegionName → Nat}

 theorem valid_size (h : Valid b shape) : b.size = shape.prod := by
  simp [Buffer.size,TensorMetadata.numel,h.shape_eq]

/-- Only active row products are normalized. No claim is made about an empty
program's prelude product or the pointer increment after the last active row. -/
 theorem active_row_stride (h : Valid b [rows,columns]) (stride : RowStride b rows columns)
    (hc : 0 < columns) (row : Nat) (hr : row < rows) : row*b.stride0 = row*columns := by
  have bound : row*columns+0 < b.size := by
    rw [valid_size h]
    simp only [List.prod_cons,List.prod_nil,Nat.mul_one,Nat.add_zero]
    exact Nat.mul_lt_mul_of_pos_right hr hc
  have contiguous := h.tensor.contiguous h.contiguous _ bound
  have actual := stride row 0 hr hc
  omega

 theorem valid_address (h : Valid b shape) (i : Nat) (hi : i < b.size) :
    b.tensor.storage.address (b.tensor.physicalIndex i) = b.pointer + 4*i := by
  have hc := h.tensor.contiguous h.contiguous i hi
  have hp := h.tensor.pointer (Nat.zero_lt_of_lt hi)
  have cap := h.tensor.inBounds i hi
  have span := b.tensor.storage.spanFits
  have base : b.tensor.storage.address b.tensor.storageOffset < 2^64 := by
    simp only [Torch.FP32.Storage.address] at *
    rw [hc] at cap
    omega
  have pointer : b.pointer = b.tensor.storage.address b.tensor.storageOffset := by
    simp only [Buffer.pointer,hp,BitVec.toNat_ofNat,Nat.mod_eq_of_lt base]
  rw [hc,pointer]
  simp [Torch.FP32.Storage.address,Nat.mul_add,Nat.add_assoc]

 theorem logical_read (h : Valid b shape) (binding : Bound pointers b)
    (memory : Nat → α) (i : Nat) (hi : i < b.size) :
    b.logical memory i = memory (address pointers (b.region,i)) := by
  rw [Buffer.logical,valid_address h i hi]
  simp [address,binding.pointer]

 theorem entry_read [Inhabited α] (h : Valid b shape) (binding : Bound pointers b)
    (memory : Nat → α) (pid i : Nat) (hi : i < b.size) :
    ((entry pointers memory pid).mem b.region i).read .fp32 = b.logical memory i := by
  rw [logical_read h binding memory i hi]
  rfl

 theorem logicalEntry_eq (inputs : List Buffer) (memory : Nat → α)
    (pointers : RegionName → Nat) (pid : Nat)
    (valid : ∀ b ∈ inputs, ∃ shape, Valid b shape)
    (binding : ∀ b ∈ inputs, Bound pointers b) :
    logicalEntry inputs memory (entry pointers memory pid) = entry pointers memory pid := by
  unfold logicalEntry
  congr 1
  funext r o
  split
  · rename_i b found
    have member := List.mem_of_find?_eq_some found
    have condition := List.find?_some found
    simp only [Bool.and_eq_true,beq_iff_eq,decide_eq_true_eq] at condition
    obtain ⟨shape,h⟩ := valid b member
    rw [logical_read h (binding b member) memory o condition.2,condition.1]
  · rfl

 theorem commit_read [Inhabited α] (pointers : RegionName → Nat) (writes : Location → Prop)
    (initial : Nat → α) (final : State α)
    (injective : ∀ a b, writes a → writes b → address pointers a = address pointers b → a = b)
    (l : Location) (hl : writes l) :
    commit pointers writes initial final (address pointers l) = (final.mem l.1 l.2).read .fp32 := by
  classical
  unfold commit
  split
  · rename_i found
    have eq := injective found.choose l found.choose_spec.1 hl found.choose_spec.2
    rw [eq]
  · rename_i absent
    exact False.elim (absent ⟨l,hl,rfl⟩)

 theorem commit_frame [Inhabited α] (pointers : RegionName → Nat) (writes : Location → Prop)
    (initial : Nat → α) (final : State α) (a : Nat)
    (outside : ∀ l, writes l → address pointers l ≠ a) :
    commit pointers writes initial final a = initial a := by
  classical
  unfold commit
  split
  · rename_i found
    exact False.elim (outside found.choose found.choose_spec.1 found.choose_spec.2)
  · rfl

 theorem commit_mem_congr [Inhabited α] (pointers : RegionName → Nat) (writes : Location → Prop)
    (initial : Nat → α) (s t : State α) (h : s.mem = t.mem) :
    commit pointers writes initial s = commit pointers writes initial t := by
  funext a
  unfold commit
  split <;> simp_all

 theorem forward_logicalEntry (c : ForwardCall) (h : ForwardDomain c pointers) (memory : Nat → α) :
    logicalEntry c.inputs memory (entry pointers memory c.pid) = entry pointers memory c.pid := by
  apply logicalEntry_eq
  · intro b hb
    simp [ForwardCall.inputs] at hb
    rcases hb with rfl|rfl|rfl
    · exact ⟨_,h.x⟩
    · exact ⟨_,h.weight⟩
    · exact ⟨_,h.bias⟩
  · intro b hb
    exact h.bound b (List.mem_append_left _ hb)

 theorem backward_logicalEntry (c : BackwardCall) (h : BackwardDomain c pointers) (memory : Nat → α) :
    logicalEntry c.inputs memory (entry pointers memory c.pid) = entry pointers memory c.pid := by
  apply logicalEntry_eq
  · intro b hb
    simp [BackwardCall.inputs] at hb
    rcases hb with rfl|rfl|rfl|rfl|rfl
    · exact ⟨_,h.x⟩
    · exact ⟨_,h.weight⟩
    · exact ⟨_,h.dy⟩
    · exact ⟨_,h.mean⟩
    · exact ⟨_,h.rstd⟩
  · intro b hb
    exact h.bound b (List.mem_append_left _ hb)

 theorem forward_outside (c : ForwardCall) (h : ForwardDomain c pointers)
    (memory : Nat → α) (l : Location) (hl : ¬ c.writes l) :
    Forward.Outside c.parameters (entry pointers memory c.pid) l.1 l.2 := by
  classical
  simp only [Forward.Outside,ForwardCall.parameters,entry,Forward.meanOffset,Forward.rstdOffset,
    Forward.outputOffset,h.yStride,h.meanStride,h.rstdStride,Nat.mul_one]
  refine ⟨?_,?_,?_⟩
  · by_cases h : l.1 = c.mean.region
    · right; intro eq; apply hl; left; exact Prod.ext h eq
    · exact Or.inl h
  · by_cases h : l.1 = c.rstd.region
    · right; intro eq; apply hl; right; left; exact Prod.ext h eq
    · exact Or.inl h
  · by_cases h : l.1 = c.y.region
    · right; intro i hi eq; apply hl; right; right; exact ⟨i.1.val,hi,Prod.ext h eq⟩
    · exact Or.inl h

 theorem backward_outside (c : BackwardCall) (h : BackwardDomain c pointers) (memory : Nat → α)
    (l : Location) (hl : ¬ c.writes l) :
    Backward.Outside c.parameters (entry pointers memory c.pid) l.1 l.2 := by
  classical
  simp only [Backward.Outside,BackwardCall.parameters,entry,Backward.start,Backward.stop,h.dwStride,h.dbStride,h.dxStride]
  refine ⟨?_,?_,?_⟩
  · by_cases h : l.1 = c.dw.region
    · right; intro i hi eq; apply hl; left; exact ⟨i.1.val,hi,Prod.ext h eq⟩
    · exact Or.inl h
  · by_cases h : l.1 = c.db.region
    · right; intro i hi eq; apply hl; right; left; exact ⟨i.1.val,hi,Prod.ext h eq⟩
    · exact Or.inl h
  · by_cases h : l.1 = c.dx.region
    · right; intro row lo hr i hi eq; apply hl; right; right
      exact ⟨row,i.1.val,lo,hr,hi,Prod.ext h eq⟩
    · exact Or.inl h

 theorem backward_input_frame (c : BackwardCall) (h : BackwardDomain c pointers)
    (memory : Nat → α) (t : State α) (b : Buffer) (hb : b ∈ c.inputs) (i : Nat) (hi : i < b.size) :
    b.logical (commit pointers c.writes memory t) i = b.logical memory i := by
  have vb : ∃ shape, Valid b shape := by
    simp [BackwardCall.inputs] at hb
    rcases hb with rfl|rfl|rfl|rfl|rfl
    · exact ⟨_,h.x⟩
    · exact ⟨_,h.weight⟩
    · exact ⟨_,h.dy⟩
    · exact ⟨_,h.mean⟩
    · exact ⟨_,h.rstd⟩
  obtain ⟨shape,valid⟩ := vb
  unfold Buffer.logical
  rw [valid_address valid i hi]
  exact commit_frame pointers c.writes memory t _ (h.physicalInputs b hb i hi)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.LayerNormLiger.KernelContract

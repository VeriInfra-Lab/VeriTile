import bench.tritonbench_g.fused_rwkv6_kernel.IO

namespace VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceCells
open VeriTile Triton FP.Structural FusedRwkv6Kernel Source
noncomputable section
attribute [local instance] Classical.propDecidable
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

def RealCell (x : Cell α) : Prop := ∃ v, x=Cell.mk .real v
@[simp] theorem realCell_mk (v : α) : RealCell (Cell.mk .real v) := ⟨v,rfl⟩
theorem realCell_read (x : Cell α) (h : RealCell x) : x=Cell.mk .real (x.read .real) := by
  rcases h with ⟨v,rfl⟩
  rfl

theorem fold_tag {ι : Type} (lanes : List ι) (active : ι → Bool)
    (name : RegionName) (address : ι → Nat) (value : ι → α)
    (m : RegionName → Nat → Cell α) (r : RegionName) (b : Nat) :
    RealCell ((lanes.foldl (fun current i => if active i then
      fun r b => if r=name ∧ b=address i then .mk .real (value i) else current r b
      else current) m) r b) ↔
      RealCell (m r b) ∨ ∃ i∈lanes,active i=Bool.true ∧ r=name ∧ b=address i := by
  induction lanes generalizing m with
  | nil => simp
  | cons i rest ih =>
    simp only [List.foldl_cons,ih]
    by_cases ai : active i=Bool.true
    · by_cases hit : r=name ∧ b=address i
      · simp [ai,hit]
      · simp [ai,hit,List.mem_cons,or_and_right,exists_or,or_assoc,or_left_comm]
    · simp [ai,List.mem_cons,or_and_right,exists_or]

theorem fold_frame {ι : Type} (lanes : List ι) (active : ι → Bool)
    (name : RegionName) (address : ι → Nat) (value : ι → Cell α)
    (m : RegionName → Nat → Cell α) (r : RegionName) (b : Nat)
    (outside : ¬∃ i∈lanes,active i=Bool.true ∧ r=name ∧ b=address i) :
    (lanes.foldl (fun current i => if active i then
      fun r b => if r=name ∧ b=address i then value i else current r b
      else current) m) r b=m r b := by
  induction lanes generalizing m with
  | nil => rfl
  | cons i rest ih =>
    rw [List.foldl_cons,ih _ (fun ⟨j,hj,h⟩ => outside ⟨j,List.mem_cons_of_mem i hj,h⟩)]
    by_cases ai : active i=Bool.true
    · have hit : ¬(r=name ∧ b=address i) := fun h => outside ⟨i,by simp,ai,h⟩
      simp [ai,hit]
    · simp [ai]

theorem row_tag (M : Algebra α) (c : Shape) (s : State α) (scale : α) (n : Nat)
    (m : Source.Cells α) (r : FusedRwkv6Kernel.Region) (b : Nat) :
    RealCell (rowWrite M c s scale n m (regionName r) b) ↔
      RealCell (m (regionName r) b) ∨ PartialWrite c (program s) n r b := by
  unfold rowWrite
  rw [fold_tag,row_hit]

theorem row_frame (M : Algebra α) (c : Shape) (s : State α) (scale : α) (n : Nat)
    (m : Source.Cells α) (r : FusedRwkv6Kernel.Region) (b : Nat)
    (outside : ¬PartialWrite c (program s) n r b) :
    rowWrite M c s scale n m (regionName r) b=m (regionName r) b := by
  apply fold_frame
  exact fun h => outside ((row_hit c s n r b).mp h)

theorem loop_tag (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (n : Nat) (r : FusedRwkv6Kernel.Region) (b : Nat) :
    RealCell (loopMemory M c s scale n (regionName r) b) ↔
      RealCell (s.mem (regionName r) b) ∨ ∃ j<n,PartialWrite c (program s) j r b := by
  induction n with
  | zero => simp [loopMemory]
  | succ n ih =>
    rw [loopMemory,row_tag,ih]
    constructor
    · rintro ((old | ⟨j,hj,hit⟩) | hit)
      · exact Or.inl old
      · exact Or.inr ⟨j,by omega,hit⟩
      · exact Or.inr ⟨n,by omega,hit⟩
    · rintro (old | ⟨j,hj,hit⟩)
      · exact Or.inl (Or.inl old)
      · by_cases eq : j=n
        · subst j;exact Or.inr hit
        · exact Or.inl (Or.inr ⟨j,by omega,hit⟩)

theorem loop_frame (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (n : Nat) (r : FusedRwkv6Kernel.Region) (b : Nat)
    (outside : ∀ j<n,¬PartialWrite c (program s) j r b) :
    loopMemory M c s scale n (regionName r) b=s.mem (regionName r) b := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [loopMemory,row_frame _ _ _ _ _ _ _ _ (outside n (by omega))]
    exact ih (fun j hj => outside j (by omega))

theorem final_tag (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (r : FusedRwkv6Kernel.Region) (b : Nat) :
    RealCell (finalMemory M c s scale (regionName r) b) ↔
      RealCell (s.mem (regionName r) b) ∨ Writes c (program s) r b := by
  by_cases hf : c.final=Bool.true
  · simp only [finalMemory,if_pos hf,terminalValue]
    rw [fold_tag,loop_tag,terminal_hit c s r b hf]
    simp only [Writes,or_assoc]
  · simp [finalMemory,hf,loop_tag,Writes,FinalWrite]

theorem final_frame (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (r : FusedRwkv6Kernel.Region) (b : Nat) (outside : ¬Writes c (program s) r b) :
    finalMemory M c s scale (regionName r) b=s.mem (regionName r) b := by
  have hl : loopMemory M c s scale c.T (regionName r) b=s.mem (regionName r) b :=
    loop_frame M c s scale c.T r b (fun n hn h => outside (Or.inl ⟨n,hn,h⟩))
  by_cases hf : c.final=Bool.true
  · simp only [finalMemory,if_pos hf]
    rw [fold_frame]
    · exact hl
    · exact fun h => outside (Or.inr ((terminal_hit c s r b hf).mp h))
  · simpa only [finalMemory,if_neg hf] using hl

theorem untracked_frame (M : Algebra α) (c : Shape) (s : State α) (scale : α)
    (r : RegionName) (b : Nat) (ho : r≠"o") (ht : r≠"ht") :
    finalMemory M c s scale r b=s.mem r b := by
  by_cases hf : c.final=Bool.true
  · simp only [finalMemory,if_pos hf]
    rw [fold_frame]
    · exact loopMemory_input M c s scale c.T r ho b
    · rintro ⟨i,_,_,eq,_⟩
      exact ht eq
  · simpa only [finalMemory,if_neg hf] using loopMemory_input M c s scale c.T r ho b

#axiomsClean final_tag
#axiomsClean final_frame
#axiomsClean untracked_frame
end
end VeriTile.Bench.Repairs.FusedRwkv6KernelScalarABIChecked.SourceCells

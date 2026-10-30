import bench.tritonbench_g.triton_argmax.Proofs.GlobalMemory

namespace VeriTile.Bench.Repairs.TritonArgmax.GlobalMemory
open Source Memory

theorem disjoint_symm {p q : Address} {n m : Nat} (apart : Disjoint p n q m) : Disjoint q m p n := by
  rcases apart with owners | before | after
  · exact Or.inl owners.symm
  · exact Or.inr (Or.inr before)
  · exact Or.inr (Or.inl after)

theorem first_mid_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (a : Nat → Word)
    (s : M) (pid query : Nat) (hp : pid < c.blocks) (hq : query < c.blocks) (different : pid ≠ query) :
    mem.read d.bytes (firstProgram mem ops c p a s pid) (address p.midValue d.bytes query)=
      mem.read d.bytes s (address p.midValue d.bytes query) := by
  have cross : Disjoint (address p.midIndex 8 pid) 8 (address p.midValue d.bytes query) d.bytes := by
    apply subword_disjoint _ _ layout.midIndex
    · exact (Nat.mul_le_mul_left _ ((Nat.succ_le_iff.mpr hp).trans layout.midFits)).trans layout.indexCapacity
    · exact (Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hq)).trans layout.midCapacity
  unfold firstProgram
  rw [mem.readFrame _ _ _ _ _ _ cross,mem.readFrame _ _ _ _ _ _ (distinct d.positive different)]

theorem first_index_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (a : Nat → Word)
    (s : M) (pid query : Nat) (hp : pid < c.blocks) (hq : query < c.midBlock) (different : pid ≠ query) :
    mem.read 8 (firstProgram mem ops c p a s pid) (address p.midIndex 8 query)=
      mem.read 8 s (address p.midIndex 8 query) := by
  have cross : Disjoint (address p.midValue d.bytes pid) d.bytes (address p.midIndex 8 query) 8 := by
    apply disjoint_symm
    apply subword_disjoint _ _ layout.midIndex
    · exact (Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hq)).trans layout.indexCapacity
    · exact (Nat.mul_le_mul_left _ (Nat.succ_le_iff.mpr hp)).trans layout.midCapacity
  unfold firstProgram
  rw [mem.readFrame _ _ _ _ _ _ (distinct (by omega : 0 < 8) different),mem.readFrame _ _ _ _ _ _ cross]

theorem fold_read {M β : Type} (step : M → Nat → M) (read : M → β) (target : Nat)
    (value : β) (jobs : List Nat) (s : M)
    (rule : ∀ state pid,pid∈jobs → read (step state pid)=if pid=target then value else read state) :
    read (jobs.foldl step s)=if target∈jobs then value else read s := by
  induction jobs generalizing s with
  | nil => simp
  | cons pid rest ih =>
    rw [List.foldl_cons,ih _ (fun state j hj => rule state j (by simp [hj])),rule s pid (by simp)]
    by_cases later : target∈rest
    · simp [later]
    · by_cases same : pid=target
      · subst pid
        simp [later]
      · have reverse : target ≠ pid := Ne.symm same
        simp [later,same,reverse]

theorem grid_values {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (s : M) (query : Nat)
    (hq : query < c.blocks) :
    mem.read d.bytes (firstGrid mem ops c p s) (address p.midValue d.bytes query)=
      stage1Value ops c (input mem p s) query := by
  have snapshots := grid_snapshot mem ops c p layout (List.range c.blocks)
    (fun pid hp => List.mem_range.mp hp) (input mem p s) s (fun _ _ => rfl)
  unfold firstGrid
  rw [snapshots.1]
  have readback := fold_read (firstProgram mem ops c p (input mem p s))
    (fun state => mem.read d.bytes state (address p.midValue d.bytes query)) query
    (stage1Value ops c (input mem p s) query) (List.range c.blocks) s
  rw [readback]
  · simp [hq]
  · intro state pid hp
    by_cases same : pid=query
    · subst pid
      rw [if_pos rfl]
      exact first_value_readback mem ops c p layout _ state query hq
    · rw [if_neg same]
      exact first_mid_frame mem ops c p layout _ state pid query (List.mem_range.mp hp) hq same

theorem grid_indices {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (s : M) (query : Nat)
    (hq : query < c.midBlock) :
    mem.read 8 (firstGrid mem ops c p s) (address p.midIndex 8 query)=
      if query < c.blocks then BitVec.ofNat 64 (stage1Index ops c (input mem p s) query)
      else mem.read 8 s (address p.midIndex 8 query) := by
  have snapshots := grid_snapshot mem ops c p layout (List.range c.blocks)
    (fun pid hp => List.mem_range.mp hp) (input mem p s) s (fun _ _ => rfl)
  unfold firstGrid
  rw [snapshots.1]
  have readback := fold_read (firstProgram mem ops c p (input mem p s))
    (fun state => mem.read 8 state (address p.midIndex 8 query)) query
    (BitVec.ofNat 64 (stage1Index ops c (input mem p s) query)) (List.range c.blocks) s
  rw [readback]
  · simp
  · intro state pid hp
    by_cases same : pid=query
    · subst pid
      rw [if_pos rfl]
      exact first_index_readback mem ops c p _ state query
    · rw [if_neg same]
      exact first_index_frame mem ops c p layout _ state pid query (List.mem_range.mp hp) hq same

theorem second_selected_congr {d : MidFormat} (ops : Scalars d) (c : GlobalConfig)
    (a b : Nat → MidWord d) (same : ∀ i,i < c.blocks → a i=b i) :
    stage2Selected ops c a=stage2Selected ops c b := by
  have loaded : stage2Loaded ops c a=stage2Loaded ops c b := by
    funext lane
    unfold stage2Loaded
    split
    · exact same _ (by assumption)
    · rfl
  simp only [stage2Selected,loaded]

def run {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) : M :=
  secondProgram mem ops c p (firstGrid mem ops c p s)

theorem global_readback {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (layout : Layout d c p) (s : M)
    (paddingInitialized : ∀ i,i < c.midBlock → mem.read 8 s (address p.midIndex 8 i)=0) :
    mem.read 8 (run mem ops c p s) (address p.output 8 0)=
      BitVec.ofNat 64 (Source.globalIndex ops c (input mem p s)) := by
  unfold run
  rw [second_readback]
  have selected := second_selected_congr ops c
    (fun i => mem.read d.bytes (firstGrid mem ops c p s) (address p.midValue d.bytes i))
    (stage1Value ops c (input mem p s)) (grid_values mem ops c p layout s)
  rw [selected,grid_indices mem ops c p layout s _ (second_gather_bound _ _ _)]
  unfold Source.globalIndex
  dsimp only
  split
  · rfl
  · rename_i inactive
    rw [paddingInitialized _ (second_gather_bound _ _ _)]
    rfl

theorem first_bytes_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (a : Nat → Word) (s : M) (pid : Nat) (point : Address)
    (mid : Disjoint (address p.midValue d.bytes pid) d.bytes point 1)
    (index : Disjoint (address p.midIndex 8 pid) 8 point 1) :
    mem.byte (firstProgram mem ops c p a s pid) point=mem.byte s point := by
  unfold firstProgram
  rw [mem.byteFrame _ _ _ _ _ index,mem.byteFrame _ _ _ _ _ mid]

theorem grid_bytes_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) (point : Address)
    (mid : ∀ pid,pid < c.blocks → Disjoint (address p.midValue d.bytes pid) d.bytes point 1)
    (index : ∀ pid,pid < c.blocks → Disjoint (address p.midIndex 8 pid) 8 point 1) :
    mem.byte (firstGrid mem ops c p s) point=mem.byte s point := by
  have frame : ∀ jobs : List Nat,(∀ pid,pid∈jobs → pid < c.blocks) → ∀ state,
      mem.byte (jobs.foldl (firstLive mem ops c p) state) point=mem.byte state point := by
    intro jobs valid
    induction jobs with
    | nil => intro state; rfl
    | cons pid rest ih =>
      intro state
      rw [List.foldl_cons,ih (fun j hj => valid j (by simp [hj]))]
      apply first_bytes_frame
      · exact mid pid (valid pid (by simp))
      · exact index pid (valid pid (by simp))
  exact frame (List.range c.blocks) (fun pid hp => List.mem_range.mp hp) s

theorem global_bytes_frame {M : Type} {d : MidFormat} (mem : Primitives M) (ops : Scalars d)
    (c : GlobalConfig) (p : Buffers) (s : M) (point : Address)
    (mid : ∀ pid,pid < c.blocks → Disjoint (address p.midValue d.bytes pid) d.bytes point 1)
    (index : ∀ pid,pid < c.blocks → Disjoint (address p.midIndex 8 pid) 8 point 1)
    (out : Disjoint (address p.output 8 0) 8 point 1) :
    mem.byte (run mem ops c p s) point=mem.byte s point := by
  unfold run
  rw [second_frame mem ops c p _ point out,grid_bytes_frame mem ops c p s point mid index]

#axiomsClean grid_values
#axiomsClean grid_indices
#axiomsClean global_readback
#axiomsClean global_bytes_frame
end VeriTile.Bench.Repairs.TritonArgmax.GlobalMemory

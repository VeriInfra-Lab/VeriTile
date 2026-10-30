import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Coverage

namespace VeriTile.Bench.Repairs.KldivOps.Backward
open VeriTile.Triton FP.Structural
open Program (Memory memoryState writeMemory)
noncomputable section
set_option maxHeartbeats 8000000

 theorem chunk_value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start col : Nat) (mem : Memory α)
    (lo : start≤col) (hi : col<start+p.block) (active : col<p.cols) :
    chunkMemory M p flag row start mem p.O (row*p.cols+col)=
      .mk .real (scalar M flag ((mem p.T (row*p.cols+col)).read .real)) := by
  let j : TileIndex [p.block] := (⟨col-start,by omega⟩,PUnit.unit)
  have hj : start+j.1.val=col := by dsimp [j]; omega
  unfold chunkMemory writeMemory
  have hm := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory
    (TileShape.allIndices [p.block]) (fun j => start+j.1.val<p.cols) p.O
    (fun j => row*p.cols+(start+j.1.val))
    (fun j => .mk .real (scalar M flag ((mem p.T (row*p.cols+(start+j.1.val))).read .real))) (memoryState mem)
  dsimp only [memoryState] at hm
  rw [←hm]
  have h := State.masked_scatter_readback p.O
    (fun j : TileIndex [p.block] => row*p.cols+(start+j.1.val))
    (fun j => .mk .real (scalar M flag ((mem p.T (row*p.cols+(start+j.1.val))).read .real)))
    (fun j => start+j.1.val<p.cols) (memoryState mem) j (by simpa only [hj] using active)
    (Program.lane_injective p row start)
  simpa only [hj] using h

theorem chunk_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (mem : Memory α) (r : RegionName) (i : Nat)
    (outside : r≠p.O ∨ ∀ j : TileIndex [p.block], start+j.1.val<p.cols → i≠row*p.cols+(start+j.1.val)) :
    chunkMemory M p flag row start mem r i=mem r i := by
  unfold chunkMemory writeMemory
  have hm := VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory
    (TileShape.allIndices [p.block]) (fun j => start+j.1.val<p.cols) p.O
    (fun j => row*p.cols+(start+j.1.val))
    (fun j => .mk .real (scalar M flag ((mem p.T (row*p.cols+(start+j.1.val))).read .real))) (memoryState mem)
  dsimp only [memoryState] at hm
  rw [←hm]
  exact State.masked_scatter_frame _ _ _ _ _ _ _
    (outside.imp id (fun h j _ hj => h j hj)) (memoryState mem)

theorem loop_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start : Nat) (mem : Memory α) (r : RegionName) (i : Nat)
    (outside : r≠p.O ∨ i < row*p.cols+start ∨ row*p.cols+p.cols ≤ i) :
    loopModel M p flag row start mem r i=mem r i := by
  rw [loopModel]
  split
  next active =>
    rw [loop_frame M p flag row (start+p.block) (chunkMemory M p flag row start mem) r i (by
      rcases outside with hr | hlo | hhi
      · exact Or.inl hr
      · exact Or.inr (Or.inl (by omega))
      · exact Or.inr (Or.inr hhi))]
    apply chunk_frame
    rcases outside with hr | hlo | hhi
    · exact Or.inl hr
    · right; intro j hj; omega
    · right; intro j hj; omega
  next => rfl
termination_by p.cols-start
decreasing_by omega

theorem loop_value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (row start col : Nat) (mem : Memory α)
    (hb : 0<p.block) (lo : start≤col) (active : col<p.cols) (ht : p.T≠p.O) :
    loopModel M p flag row start mem p.O (row*p.cols+col)=
      .mk .real (scalar M flag ((mem p.T (row*p.cols+col)).read .real)) := by
  rw [loopModel,dif_pos ⟨hb,by omega⟩]
  by_cases hit : col<start+p.block
  · rw [loop_frame M p flag row (start+p.block)
      (chunkMemory M p flag row start mem) p.O (row*p.cols+col) (Or.inr (Or.inl (by omega)))]
    exact chunk_value M p flag row start col mem lo hit active
  · rw [loop_value M p flag row (start+p.block) col
      (chunkMemory M p flag row start mem) hb (by omega) active ht]
    rw [chunk_frame M p flag row start mem p.T (row*p.cols+col) (Or.inl ht)]
termination_by p.cols-start
decreasing_by omega

theorem grid_frame {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (jobs : List Nat) (mem : Memory α) (r : RegionName) (i : Nat)
    (outside : ∀ row∈jobs,r≠p.O ∨ i < row*p.cols ∨ row*p.cols+p.cols ≤ i) :
    gridModel M p flag jobs mem r i=mem r i := by
  induction jobs generalizing mem with
  | nil => rfl
  | cons row rest ih =>
    change gridModel M p flag rest (loopModel M p flag row 0 mem) r i=mem r i
    rw [ih _ (by intro x hx; exact outside x (by simp [hx]))]
    exact loop_frame M p flag row 0 mem r i (by simpa only [Nat.add_zero] using outside row (by simp))

theorem grid_value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (jobs : List Nat) (mem : Memory α) (row col : Nat)
    (member : row∈jobs) (hb : 0<p.block) (hc : col<p.cols) (ht : p.T≠p.O) :
    gridModel M p flag jobs mem p.O (row*p.cols+col)=
      .mk .real (scalar M flag ((mem p.T (row*p.cols+col)).read .real)) := by
  induction jobs generalizing mem with
  | nil => simp at member
  | cons first rest ih =>
    change gridModel M p flag rest (loopModel M p flag first 0 mem) p.O (row*p.cols+col)=_
    by_cases later : row∈rest
    · rw [ih _ later,loop_frame M p flag first 0 mem p.T (row*p.cols+col) (Or.inl ht)]
    · have eq : row=first := by simpa only [List.mem_cons,later,or_false] using member
      subst first
      rw [grid_frame M p flag rest _ p.O (row*p.cols+col) (by
        intro other ho
        right
        by_contra h
        have lo : other*p.cols≤row*p.cols+col := by omega
        have hi : row*p.cols+col<other*p.cols+p.cols := by omega
        have he := Program.coordinate_row p.cols row col other hc lo hi
        exact later (he ▸ ho))]
      exact loop_value M p flag row 0 col mem hb (by omega) hc ht

theorem grid_tail {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (mem : Memory α) (i : Nat) (tail : p.rows*p.cols ≤ i) :
    gridModel M p flag (List.range p.rows) mem p.O i=mem p.O i := by
  apply grid_frame
  intro row hrow
  have hm := Nat.mul_le_mul_right p.cols (show row+1≤p.rows by have := List.mem_range.mp hrow; omega)
  simp only [Nat.add_mul,Nat.one_mul] at hm
  exact Or.inr (Or.inr (by omega))

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Backward

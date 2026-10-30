import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Frames
import bench.repairs.tritonbench_g.kldiv_ops.Proofs.Backward

namespace VeriTile.Bench.Repairs.KldivOps.Program
open VeriTile.Triton FP.Structural Tile
noncomputable section
set_option maxHeartbeats 8000000

 theorem lane_injective (p : Parameters) (row start : Nat) :
    Function.Injective (fun j : TileIndex [p.block] => row*p.cols+(start+j.1.val)) := by
  intro i j h
  dsimp only at h
  exact Prod.ext (Fin.ext (by omega)) (Subsingleton.elim _ _)

theorem chunk_value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (eps : α) (row start col : Nat) (s : State α)
    (lo : start≤col) (hi : col<start+p.block) (active : col<p.cols) :
    chunkMemory M p flag 0 row start eps s p.O (row*p.cols+col)=
      .mk .real (scalar M flag eps ((s.mem p.Y (row*p.cols+col)).read .real)
        ((s.mem p.T (row*p.cols+col)).read .real)) := by
  let j : TileIndex [p.block] := (⟨col-start,by omega⟩,PUnit.unit)
  have hj : start+j.1.val=col := by dsimp [j]; omega
  unfold chunkMemory
  rw [if_pos rfl,←VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs.masked_scatter_memory]
  have h := State.masked_scatter_readback p.O
    (fun j : TileIndex [p.block] => row*p.cols+(start+j.1.val))
    (fun j => .mk .real (lane M p flag eps row start s j))
    (fun j => start+j.1.val<p.cols) s j (by simpa only [hj] using active) (lane_injective p row start)
  simpa only [hj,lane,active,if_true] using h

theorem loop_before {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (red row : Nat) (eps : α) (start : Nat) (data : Memory α×α) (i : Nat)
    (before : i<row*p.cols+start) :
    (loopModel M p flag red row eps start data).1 p.O i=data.1 p.O i := by
  rw [loopModel]
  split
  next active =>
    rw [loop_before M p flag red row eps (start+p.block) (advance M p flag red row start eps data) i (by omega)]
    apply chunk_frame
    right
    intro j hj
    omega
  next => rfl
termination_by p.cols-start
decreasing_by omega

theorem loop_value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (eps : α) (row start col : Nat) (data : Memory α×α)
    (hb : 0<p.block) (lo : start≤col) (active : col<p.cols)
    (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    (loopModel M p flag 0 row eps start data).1 p.O (row*p.cols+col)=
      .mk .real (scalar M flag eps ((data.1 p.Y (row*p.cols+col)).read .real)
        ((data.1 p.T (row*p.cols+col)).read .real)) := by
  rw [loopModel,dif_pos ⟨hb,by omega⟩]
  by_cases hit : col<start+p.block
  · rw [loop_before M p flag 0 row eps (start+p.block)
      (advance M p flag 0 row start eps data) (row*p.cols+col) (by omega)]
    exact chunk_value M p flag eps row start col (memoryState data.1) lo hit active
  · rw [loop_value M p flag eps row (start+p.block) col
      (advance M p flag 0 row start eps data) hb (by omega) active hy ht]
    have fy := chunk_frame M p flag 0 row start eps (memoryState data.1) p.Y (row*p.cols+col) (Or.inl hy)
    have ft := chunk_frame M p flag 0 row start eps (memoryState data.1) p.T (row*p.cols+col) (Or.inl ht)
    change Cell.mk .real (scalar M flag eps
      ((chunkMemory M p flag 0 row start eps (memoryState data.1) p.Y (row*p.cols+col)).read .real)
      ((chunkMemory M p flag 0 row start eps (memoryState data.1) p.T (row*p.cols+col)).read .real))=_
    rw [fy,ft]
    rfl
termination_by p.cols-start
decreasing_by omega

theorem row_value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (eps : α) (row col : Nat) (mem : Memory α) (hb : 0<p.block) (hc : col<p.cols)
    (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    rowModel M p flag 0 row eps mem p.O (row*p.cols+col)=
      .mk .real (scalar M flag eps ((mem p.Y (row*p.cols+col)).read .real)
        ((mem p.T (row*p.cols+col)).read .real)) := by
  exact loop_value M p flag eps row 0 col (mem,zero M) hb (by omega) hc hy ht

theorem coordinate_row (cols row col other : Nat) (hc : col<cols)
    (lo : other*cols≤row*cols+col) (hi : row*cols+col<other*cols+cols) : other=row := by
  by_contra neq
  rcases lt_or_gt_of_ne neq with lt | gt
  · have hm := Nat.mul_le_mul_right cols (show other+1≤row by omega)
    simp only [Nat.add_mul,Nat.one_mul] at hm
    omega
  · have hm := Nat.mul_le_mul_right cols (show row+1≤other by omega)
    simp only [Nat.add_mul,Nat.one_mul] at hm
    omega

theorem grid_value {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (flag : Bool)
    (eps : α) (jobs : List Nat) (mem : Memory α) (row col : Nat)
    (member : row∈jobs) (hb : 0<p.block) (hc : col<p.cols)
    (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    gridModel M p flag 0 eps jobs mem p.O (row*p.cols+col)=
      .mk .real (scalar M flag eps ((mem p.Y (row*p.cols+col)).read .real)
        ((mem p.T (row*p.cols+col)).read .real)) := by
  induction jobs generalizing mem with
  | nil => simp at member
  | cons first rest ih =>
    change gridModel M p flag 0 eps rest (rowModel M p flag 0 first eps mem) p.O (row*p.cols+col)=_
    by_cases later : row∈rest
    · rw [ih _ later]
      have fy := row_frame M p flag 0 first eps mem p.Y (row*p.cols+col) (by simp [Writes,hy])
      have ft := row_frame M p flag 0 first eps mem p.T (row*p.cols+col) (by simp [Writes,ht])
      rw [fy,ft]
    · have eq : row=first := by simpa only [List.mem_cons,later,or_false] using member
      subst first
      rw [grid_frame M p flag 0 eps rest _ p.O (row*p.cols+col) (by
        intro other ho hw
        have hh : other*p.cols≤row*p.cols+col ∧ row*p.cols+col<other*p.cols+p.cols := by simpa [Writes] using hw
        have he := coordinate_row p.cols row col other hc hh.1 hh.2
        exact later (he ▸ ho))]
      exact row_value M p flag eps row col mem hb hc hy ht

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.Program

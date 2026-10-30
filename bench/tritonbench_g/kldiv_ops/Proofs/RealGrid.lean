import bench.tritonbench_g.kldiv_ops.Proofs.Real
import bench.tritonbench_g.kldiv_ops.Proofs.Coverage

namespace VeriTile.Bench.Repairs.KldivOps.RealExecution
open VeriTile.Triton FP.Structural
open Program Tile
open VeriTile.Bench.TritonBenchG.KldivOps.FullMath (prefixLoss prefixLoss_succ prefixLoss_zero prefixLoss_eq_rowLoss)
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 8000000

 theorem reduced_loop (p : Parameters) (flag : Bool) (red row : Nat) (eps : ℝ)
    (mem : Memory ℝ) (chunk : Nat) (hb : 0<p.block) (hr : red≠0) :
    loopModel algebra p flag red row eps (chunk*p.block)
      (mem,prefixLoss flag eps (fun i => (mem p.Y (row*p.cols+i)).read .real)
        (fun i => (mem p.T (row*p.cols+i)).read .real) p.cols p.block chunk)=
      (mem,rowLoss flag eps (fun i => (mem p.Y (row*p.cols+i)).read .real)
        (fun i => (mem p.T (row*p.cols+i)).read .real) p.cols) := by
  rw [loopModel]
  split
  next active =>
    have hm : advance algebra p flag red row (chunk*p.block) eps
        (mem,prefixLoss flag eps (fun i => (mem p.Y (row*p.cols+i)).read .real)
          (fun i => (mem p.T (row*p.cols+i)).read .real) p.cols p.block chunk)=
        (mem,prefixLoss flag eps (fun i => (mem p.Y (row*p.cols+i)).read .real)
          (fun i => (mem p.T (row*p.cols+i)).read .real) p.cols p.block (chunk+1)) := by
      simp only [advance,chunkMemory,hr,if_false,nextCarry]
      rw [padded_chunk_real]
      rw [prefixLoss_succ]
      rfl
    rw [hm,show chunk*p.block+p.block=(chunk+1)*p.block by ring]
    exact reduced_loop p flag red row eps mem (chunk+1) hb hr
  next inactive =>
    have cover : p.cols≤chunk*p.block := by omega
    rw [prefixLoss_eq_rowLoss _ _ _ _ _ _ _ cover]
termination_by p.cols-chunk*p.block
decreasing_by
  simp only [Nat.add_mul,Nat.one_mul]
  omega

theorem reduced_row (p : Parameters) (flag : Bool) (red row : Nat) (eps : ℝ)
    (mem : Memory ℝ) (hb : 0<p.block) (hr : red≠0) :
    rowModel algebra p flag red row eps mem p.O row=
      .mk .real (rowLoss flag eps (fun i => (mem p.Y (row*p.cols+i)).read .real)
        (fun i => (mem p.T (row*p.cols+i)).read .real) p.cols) := by
  have h := reduced_loop p flag red row eps mem 0 hb hr
  simp only [Nat.zero_mul,prefixLoss_zero] at h
  unfold rowModel
  rw [if_neg hr]
  change writeMemory (loopModel algebra p flag red row eps 0 (mem,0)).1 p.O row
    (.mk .real (loopModel algebra p flag red row eps 0 (mem,0)).2) p.O row=_
  rw [h]
  simp [writeMemory]

theorem reduced_grid (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ)
    (jobs : List Nat) (mem : Memory ℝ) (row : Nat) (member : row∈jobs)
    (hb : 0<p.block) (hr : red≠0) (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    gridModel algebra p flag red eps jobs mem p.O row=
      .mk .real (rowLoss flag eps (fun i => (mem p.Y (row*p.cols+i)).read .real)
        (fun i => (mem p.T (row*p.cols+i)).read .real) p.cols) := by
  induction jobs generalizing mem with
  | nil => simp at member
  | cons first rest ih =>
    change gridModel algebra p flag red eps rest (rowModel algebra p flag red first eps mem) p.O row=_
    by_cases later : row∈rest
    · rw [ih _ later]
      congr 2
      · funext i
        rw [row_frame algebra p flag red first eps mem p.Y (row*p.cols+i) (by simp [Writes,hy])]
      · funext i
        rw [row_frame algebra p flag red first eps mem p.T (row*p.cols+i) (by simp [Writes,ht])]
    · have eq : row=first := by simpa only [List.mem_cons,later,or_false] using member
      subst first
      rw [grid_frame algebra p flag red eps rest _ p.O row (by
        intro other ho hw
        have eq : row=other := by simpa [Writes,hr] using hw
        exact later (eq ▸ ho))]
      exact reduced_row p flag red row eps mem hb hr

/-- All rows and active elementwise cells of the actual complete repaired
source have the independent logical Real values, including empty dimensions. -/
theorem complete_grid (p : Parameters) (flag : Bool) (red : Nat) (eps : ℝ)
    (s : State ℝ) (hb : 0<p.block) (hy : p.Y≠p.O) (ht : p.T≠p.O) :
    ∃ final, grid algebra p flag red eps s=some final ∧
      (∀ row,row<p.rows → ∀ col,col<p.cols → red=0 →
        (final.mem p.O (row*p.cols+col)).read .real=
          loss flag eps ((s.mem p.Y (row*p.cols+col)).read .real)
            ((s.mem p.T (row*p.cols+col)).read .real)) ∧
      (∀ row,row<p.rows → red≠0 → (final.mem p.O row).read .real=
        rowLoss flag eps (fun i => (s.mem p.Y (row*p.cols+i)).read .real)
          (fun i => (s.mem p.T (row*p.cols+i)).read .real) p.cols) := by
  obtain ⟨final,he,hm⟩ := grid_complete algebra p flag red eps s hb
  refine ⟨final,he,?_,?_⟩
  · intro row hrow col hcol hz
    subst red
    rw [hm,grid_value algebra p flag eps (List.range p.rows) s.mem row col
      (List.mem_range.mpr hrow) hb hcol hy ht]
    simpa using scalar_real flag eps ((s.mem p.Y (row*p.cols+col)).read .real)
      ((s.mem p.T (row*p.cols+col)).read .real)
  · intro row hrow hr
    rw [hm,reduced_grid p flag red eps (List.range p.rows) s.mem row (List.mem_range.mpr hrow) hb hr hy ht]
    simp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivOps.RealExecution

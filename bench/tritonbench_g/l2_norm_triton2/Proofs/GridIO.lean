import bench.tritonbench_g.l2_norm_triton2.IO
import bench.tritonbench_g.l2_norm_triton2.Proofs.AccessSafety
import bench.tritonbench_g.l2_norm_triton2.Proofs.Kernel
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.L2NormTriton2.GridIO
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 24000

theorem launch_initial {α : Type} [Inhabited α] (A : Algebra α)
    (N B row : Nat) (eps : α) (s : State α) :
    (launch A N B row eps).initial s = Tile.rowState row eps s := rfl

theorem launch_run {α : Type} [Inhabited α] (A : Algebra α)
    (N B row : Nat) (eps : α) (s : State α) :
    (launch A N B row eps).run s =
      FP.Structural.exec A (backward N B) (Tile.rowState row eps s) := rfl

theorem candidate_run {α : Type} [Inhabited α] (A : Algebra α)
    (N B row : Nat) (eps : α) (s : State α) :
    (candidateLaunch A N B row eps).run s = (launch A N B row eps).run s :=
  Kernel.commuted_execution A N B row eps s

theorem schedule_succ {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :
    schedule A (rows + 1) N B eps = schedule A rows N B eps ++ [launch A N B rows eps] := by
  simp [schedule, List.range_succ, List.map_append]

theorem candidateSchedule_succ {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α) :
    candidateSchedule A (rows + 1) N B eps =
      candidateSchedule A rows N B eps ++ [candidateLaunch A N B rows eps] := by
  simp [candidateSchedule, List.range_succ, List.map_append]

def Outside (rows N : Nat) (region : RegionName) (offset : Nat) : Prop :=
  region ≠ "DX" ∨ rows * N ≤ offset

theorem prior_row (N rows row col : Nat) (hr : row < rows) (hc : col < N) :
    row * N + col < rows * N := by
  have h := Nat.mul_le_mul_right N (Nat.succ_le_of_lt hr)
  simp only [Nat.succ_mul] at h
  omega

/-- Successful execution of the entire row grid, source-derived scalar values,
and the exact exterior cell frame. Inputs remain those of the original state. -/
theorem grid_execution {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (s : State α) :
    ∃ t, runSchedule (schedule A rows N B eps) s = some t ∧
      (∀ row col, row < rows → col < N → col < B →
        t.mem "DX" (row * N + col) = .mk .real
          (Tile.scalar A N B eps (Tile.inputs s "X" N row) (Tile.inputs s "DY" N row) col)) ∧
      (∀ r o, Outside rows N r o → t.mem r o = s.mem r o) := by
  induction rows with
  | zero =>
    refine ⟨s, rfl, ?_, fun _ _ _ => rfl⟩
    intro row col h
    omega
  | succ rows ih =>
    obtain ⟨middle, previous, values, frame⟩ := ih
    let final := (Tile.sourceRun A N B rows eps middle).val
    have last : (launch A N B rows eps).run middle = some final :=
      (Tile.sourceRun A N B rows eps middle).property
    refine ⟨final, ?_, ?_, ?_⟩
    · rw [schedule_succ, runSchedule_append, previous, Option.bind_some, runSchedule_singleton]
      exact last
    · intro row col hr hc hb
      by_cases same : row = rows
      · subst row
        have inputEq : Tile.inputs middle "X" N rows = Tile.inputs s "X" N rows := by
          funext j
          simp only [Tile.inputs, frame "X" (rows * N + j) (Or.inl (by decide))]
        have gradientEq : Tile.inputs middle "DY" N rows = Tile.inputs s "DY" N rows := by
          funext j
          simp only [Tile.inputs, frame "DY" (rows * N + j) (Or.inl (by decide))]
        rw [Tile.readback A N B rows eps middle (⟨col, hb⟩, PUnit.unit) hc, inputEq, gradientEq]
      · have before : row < rows := by omega
        have unchanged := Tile.frame A N B rows eps middle "DX" (row * N + col)
          (Or.inr (fun j active => by
            have bound := prior_row N rows row col before hc
            omega))
        rw [unchanged, values row col before hc hb]
    · intro r o outside
      have unchanged : final.mem r o = middle.mem r o := by
        apply Tile.frame
        rcases outside with different | bound
        · exact Or.inl different
        · right
          intro j active
          simp only [Nat.succ_mul] at bound
          omega
      rw [unchanged]
      apply frame r o
      exact outside.imp id (fun h => (Nat.mul_le_mul_right N (Nat.le_succ rows)).trans h)

theorem candidate_execution {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (s : State α) :
    runSchedule (candidateSchedule A rows N B eps) s = runSchedule (schedule A rows N B eps) s := by
  induction rows with
  | zero => rfl
  | succ rows ih =>
    rw [candidateSchedule_succ, schedule_succ, runSchedule_append, runSchedule_append, ih]
    congr 1
    funext before
    simp only [runSchedule_singleton, candidate_run]

private theorem schedule_safe_of_all {α : Type} [Inhabited α]
    (bounds : RegionName → Nat) (jobs : List (Invocation α))
    (safe : ∀ job ∈ jobs, ∀ s, job.Safe bounds s) (s : State α) : ScheduleSafe bounds jobs s := by
  induction jobs generalizing s with
  | nil => trivial
  | cons job rest ih =>
    exact ⟨safe job (by simp) s,
      fun t _ => ih (fun next h => safe next (by simp [h])) t⟩

theorem schedule_safe {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (bounds : RegionName → Nat)
    (input : ∀ row col, row < rows → col < N → row * N + col < bounds "X")
    (gradient : ∀ row col, row < rows → col < N → row * N + col < bounds "DY")
    (output : ∀ row col, row < rows → col < N → row * N + col < bounds "DX") (s : State α) :
    ScheduleSafe bounds (schedule A rows N B eps) s := by
  apply schedule_safe_of_all
  intro job member before
  obtain ⟨row, hr, rfl⟩ := List.mem_map.mp member
  have rowBound := List.mem_range.mp hr
  exact Tile.source_traceSafe A N B row eps bounds before
    (fun j hj => input row j.val rowBound hj)
    (fun j hj => gradient row j.val rowBound hj) (fun j hj => output row j.val rowBound hj)

theorem candidate_safe {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (bounds : RegionName → Nat)
    (input : ∀ row col, row < rows → col < N → row * N + col < bounds "X")
    (gradient : ∀ row col, row < rows → col < N → row * N + col < bounds "DY")
    (output : ∀ row col, row < rows → col < N → row * N + col < bounds "DX") (s : State α) :
    ScheduleSafe bounds (candidateSchedule A rows N B eps) s := by
  apply schedule_safe_of_all
  intro job member before
  obtain ⟨row, hr, rfl⟩ := List.mem_map.mp member
  have rowBound := List.mem_range.mp hr
  exact Tile.commuted_traceSafe A N B row eps bounds before
    (fun j hj => input row j.val rowBound hj)
    (fun j hj => gradient row j.val rowBound hj) (fun j hj => output row j.val rowBound hj)


/-- Every declared logical input and output window supplies the actual source
bounds for every row in the complete schedule. -/
theorem io_bounds {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (vals : Inputs α rows N)
    (bounds : RegionName → Nat) (bounded : (gridIO A rows N B eps).InBounds bounds vals) :
    (∀ row col, row < rows → col < N → row*N+col < bounds "X") ∧
    (∀ row col, row < rows → col < N → row*N+col < bounds "DY") ∧
    (∀ row col, row < rows → col < N → row*N+col < bounds "DX") := by
  refine ⟨?_,?_,?_⟩
  · intro row col hr hc
    exact bounded.2.1 0 (⟨row,hr⟩,⟨col,hc⟩,PUnit.unit) trivial
  · intro row col hr hc
    exact bounded.2.1 1 (⟨row,hr⟩,⟨col,hc⟩,PUnit.unit) trivial
  · intro row col hr hc
    exact bounded.2.2.1 0 (⟨row,hr⟩,⟨col,hc⟩,PUnit.unit) trivial

/-- The full logical output rectangle is exactly the grid's writable interval. -/
theorem io_outside {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α)
    (vals : Inputs α rows N) (r : RegionName) (offset : Nat)
    (outside : (gridIO A rows N B eps).Outside vals r offset) : Outside rows N r offset := by
  by_cases other : r ≠ "DX"
  · exact Or.inl other
  have same : r="DX" := by simpa using other
  subst r
  right
  by_contra notBound
  have small : offset < rows*N := by omega
  have npos : 0<N := by
    by_contra h
    have hn : N=0 := by omega
    simp [hn] at small
  have rowBound : offset/N < rows := (Nat.div_lt_iff_lt_mul npos).mpr small
  have colBound : offset%N < N := Nat.mod_lt _ npos
  have excluded := outside.1 0 (⟨offset/N,rowBound⟩,⟨offset%N,colBound⟩,PUnit.unit) trivial
  change ("DX" : RegionName) ≠ "DX" ∨ offset ≠ offset/N*N+offset%N at excluded
  rcases excluded with different | different
  · exact different rfl
  · apply different
    simpa only [Nat.mul_comm] using (Nat.div_add_mod offset N).symm

/-- The source's real output is the independent full-row gradient; only block
coverage connects the finite logical columns to physical source lanes. -/
theorem real_implements (rows N B : Nat) (cover : N≤B) (eps : ℝ) :
    (gridIO RealExecution.algebra rows N B eps).Implements (mathematical rows N eps) := by
  intro vals _ bounds bounded before binding
  obtain ⟨bx,bd,bo⟩ := io_bounds RealExecution.algebra rows N B eps vals bounds bounded
  obtain ⟨after,ran,values,frame⟩ := grid_execution RealExecution.algebra rows N B eps before
  refine ⟨schedule_safe _ _ _ _ _ _ bx bd bo before,after,ran,?_,?_⟩
  · intro o j _
    have ho : o=0 := Fin.eq_zero o
    subst o
    rcases j with ⟨row,col,⟨⟩⟩
    have hx (i : Nat) (hi : i<N) :
        Tile.inputs before "X" N row.val i = rowValues vals Bool.false row i := by
      have h := binding 0 (row,⟨i,hi⟩,PUnit.unit) trivial
      simpa [Tile.inputs,rowValues,hi,mapValue] using h
    have hd (i : Nat) (hi : i<N) :
        Tile.inputs before "DY" N row.val i = rowValues vals Bool.true row i := by
      have h := binding 1 (row,⟨i,hi⟩,PUnit.unit) trivial
      simpa [Tile.inputs,rowValues,hi,mapValue] using h
    have actual := values row.val col.val row.isLt col.isLt (lt_of_lt_of_le col.isLt cover)
    rw [Tile.scalar_congr _ N B eps _ _ _ _ hx hd col.val col.isLt,
      RealExecution.scalar_real N B eps _ _ col.val cover] at actual
    change mapCell (id : ℝ → ℝ) (after.mem "DX" (row.val*N+col.val)) = _
    rw [actual]
    rfl
  · intro r offset outside
    exact frame r offset (io_outside _ _ _ _ _ vals r offset outside)

/-- The existing integer-address product commute preserves every observed cell
of both successful and safe full-grid executions for every primitive algebra.
Floating operations, exceptional words and the runtime epsilon are unchanged. -/
theorem equivalence {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    (gridIO A rows N B eps).Equivalent (candidate A rows N B eps) := by
  intro vals _ bounds bounded before _
  obtain ⟨bx,bd,bo⟩ := io_bounds A rows N B eps vals bounds bounded
  obtain ⟨after,ran,_,frame⟩ := grid_execution A rows N B eps before
  have rightRan : runSchedule (candidateSchedule A rows N B eps) before=some after := by
    rw [candidate_execution]
    exact ran
  have framed : (gridIO A rows N B eps).Frame vals before after := by
    intro r offset outside
    exact frame r offset (io_outside _ _ _ _ _ vals r offset outside)
  exact ⟨schedule_safe A rows N B eps bounds bx bd bo before,
    candidate_safe A rows N B eps bounds bx bd bo before,
    after,after,ran,rightRan,fun _ _ _ => rfl,framed,framed⟩

#axiomsClean real_implements
#axiomsClean equivalence

#axiomsClean grid_execution
#axiomsClean candidate_execution
#axiomsClean schedule_safe
#axiomsClean candidate_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton2.GridIO

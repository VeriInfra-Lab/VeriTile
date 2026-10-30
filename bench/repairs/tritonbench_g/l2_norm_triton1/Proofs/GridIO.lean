import bench.repairs.tritonbench_g.l2_norm_triton1.IO
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.AccessSafety
import bench.repairs.tritonbench_g.l2_norm_triton1.Proofs.Kernel
import VeriTile.Triton.Float.ProgramEquivalence

namespace VeriTile.Bench.Repairs.L2NormTriton1.GridIO
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
      FP.Structural.exec A (source N B) (Tile.rowState row eps s) := rfl

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
  region ≠ "Y" ∨ rows * N ≤ offset

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
        t.mem "Y" (row * N + col) = .mk .real
          (Tile.scalar A N B eps (Tile.inputs s N row) col)) ∧
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
        have inputEq : Tile.inputs middle N rows = Tile.inputs s N rows := by
          funext j
          simp only [Tile.inputs, frame "X" (rows * N + j) (Or.inl (by decide))]
        rw [Tile.readback A N B rows eps middle (⟨col, hb⟩, PUnit.unit) hc, inputEq]
      · have before : row < rows := by omega
        have unchanged := Tile.frame A N B rows eps middle "Y" (row * N + col)
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
    (output : ∀ row col, row < rows → col < N → row * N + col < bounds "Y") (s : State α) :
    ScheduleSafe bounds (schedule A rows N B eps) s := by
  apply schedule_safe_of_all
  intro job member before
  obtain ⟨row, hr, rfl⟩ := List.mem_map.mp member
  have rowBound := List.mem_range.mp hr
  exact Tile.source_traceSafe A N B row eps bounds before
    (fun j hj => input row j.val rowBound hj) (fun j hj => output row j.val rowBound hj)

theorem candidate_safe {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (bounds : RegionName → Nat)
    (input : ∀ row col, row < rows → col < N → row * N + col < bounds "X")
    (output : ∀ row col, row < rows → col < N → row * N + col < bounds "Y") (s : State α) :
    ScheduleSafe bounds (candidateSchedule A rows N B eps) s := by
  apply schedule_safe_of_all
  intro job member before
  obtain ⟨row, hr, rfl⟩ := List.mem_map.mp member
  have rowBound := List.mem_range.mp hr
  exact Tile.commuted_traceSafe A N B row eps bounds before
    (fun j hj => input row j.val rowBound hj) (fun j hj => output row j.val rowBound hj)

/-- Bounds on every declared logical input/output imply bounds for every
actual source access in all row invocations. -/
theorem io_safe {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (vals : Inputs α rows N)
    (bounds : RegionName → Nat) (covered : (gridIO A rows N B eps).InBounds bounds vals)
    (s : State α) : ScheduleSafe bounds (schedule A rows N B eps) s := by
  apply schedule_safe
  · intro row col hr hc
    exact covered.2.1 (0 : Fin 1) (⟨row, hr⟩, ⟨col, hc⟩, PUnit.unit) trivial
  · intro row col hr hc
    exact covered.2.2.1 (0 : Fin 1) (⟨row, hr⟩, ⟨col, hc⟩, PUnit.unit) trivial

theorem candidate_io_safe {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (vals : Inputs α rows N)
    (bounds : RegionName → Nat) (covered : (gridIO A rows N B eps).InBounds bounds vals)
    (s : State α) : ScheduleSafe bounds (candidateSchedule A rows N B eps) s := by
  apply candidate_safe
  · intro row col hr hc
    exact covered.2.1 (0 : Fin 1) (⟨row, hr⟩, ⟨col, hc⟩, PUnit.unit) trivial
  · intro row col hr hc
    exact covered.2.2.1 (0 : Fin 1) (⟨row, hr⟩, ⟨col, hc⟩, PUnit.unit) trivial

theorem row_binding {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) (vals : Inputs α rows N) (s : State α)
    (pins : (gridIO A rows N B eps).InputBinding vals s)
    (row : Fin rows) (col : Nat) (hc : col < N) :
    Tile.inputs s N row.val col = rowValues vals row col := by
  simpa [Tile.inputs, rowValues, hc, mapValue] using
    pins (0 : Fin 1) (row, ⟨col, hc⟩, PUnit.unit) trivial

/-- The declared dense output coordinates cover exactly the full grid region. -/
theorem outside_of_io {α : Type} (A : Algebra α) (rows N B : Nat) (eps : α)
    (vals : Inputs α rows N) (r : RegionName) (o : Nat)
    (outside : (gridIO A rows N B eps).Outside vals r o) : Outside rows N r o := by
  by_cases region : r = "Y"
  · right
    by_contra notBound
    have below : o < rows * N := by omega
    have npos : 0 < N := by nlinarith
    have hr : o / N < rows := (Nat.div_lt_iff_lt_mul npos).mpr below
    have hc : o % N < N := Nat.mod_lt o npos
    have unequal := (outside.1 (0 : Fin 1) (⟨o / N, hr⟩, ⟨o % N, hc⟩, PUnit.unit)
      trivial).resolve_left (not_not.mpr region)
    apply unequal
    change o = o / N * N + o % N
    simpa only [Nat.mul_comm] using (Nat.div_add_mod o N).symm
  · exact Or.inl region

theorem formula_congr (N : Nat) (eps : ℝ) (xs ys : Nat → ℝ) (col : Nat)
    (hc : col < N) (same : ∀ j, j < N → xs j = ys j) :
    RealExecution.formula N eps xs col = formula N eps ys col := by
  unfold RealExecution.formula formula
  rw [same col hc]
  congr 3
  apply Finset.sum_congr rfl
  intro j hj
  rw [same j (Finset.mem_range.mp hj)]

/-- Full-grid public correctness assembled from source safety and the existing
source row results. Only logical input arrays occur in its mathematical target. -/
theorem real_implements (rows N B : Nat) (cover : N ≤ B) (eps : ℝ) :
    (gridIO RealExecution.algebra rows N B eps).Implements (mathematical rows N eps) := by
  apply ProgramIO.Implements.intro
  · intro vals bounds covered initial _
    exact io_safe _ rows N B eps vals bounds covered initial
  · intro vals initial pins
    obtain ⟨final, ran, outputs, frame⟩ := grid_execution RealExecution.algebra rows N B eps initial
    refine ⟨final, ran, ?_, ?_⟩
    · intro o
      refine Fin.cases ?_ (fun i => i.elim0) o
      intro j _
      have value := outputs j.1.val j.2.1.val j.1.isLt j.2.1.isLt
        (lt_of_lt_of_le j.2.1.isLt cover)
      change mapCell id (final.mem "Y" (j.1.val * N + j.2.1.val)) = _
      rw [value]
      change Cell.mk .real (Tile.scalar _ _ _ _ _ _) = Cell.mk .real _
      rw [RealExecution.scalar_real N B eps _ j.2.1.val cover]
      apply congrArg (Cell.mk .real)
      exact formula_congr N eps _ _ j.2.1.val j.2.1.isLt
        (row_binding _ rows N B eps vals initial pins j.1)
    · intro r o outside
      exact frame r o (outside_of_io _ rows N B eps vals r o outside)

/-- Full schedules agree with identical complete states for any primitive
algebra and epsilon payload. There is no floating transformation premise. -/
theorem equivalence {α : Type} [Inhabited α] (A : Algebra α)
    (rows N B : Nat) (eps : α) :
    (gridIO A rows N B eps).Equivalent (candidate A rows N B eps) := by
  intro vals _ bounds covered initial _
  refine ⟨io_safe A rows N B eps vals bounds covered initial,
    candidate_io_safe A rows N B eps vals bounds covered initial, ?_⟩
  obtain ⟨final, ran, _, frame⟩ := grid_execution A rows N B eps initial
  have framed : (gridIO A rows N B eps).Frame vals initial final :=
    fun r o outside => frame r o (outside_of_io A rows N B eps vals r o outside)
  refine ⟨final, final, ran, ?_, ?_, framed, framed⟩
  · change runSchedule (candidateSchedule A rows N B eps) initial = some final
    rw [candidate_execution, ran]
  · intro o j _
    rfl

#axiomsClean grid_execution
#axiomsClean candidate_execution
#axiomsClean schedule_safe
#axiomsClean candidate_safe
#axiomsClean real_implements
#axiomsClean equivalence
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.L2NormTriton1.GridIO

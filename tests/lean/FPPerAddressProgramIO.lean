import VeriTile.Triton.Float.PerAddressProgramIO
import Mathlib.Tactic.FinCases

namespace VeriTile.Triton.FP.Structural.PerAddressRegression
noncomputable section

def operation (addOne : Bool) (x : Int) : Int := if addOne then x + 1 else x * 2

def expression (addOne : Bool) (r : RegionName) : Op .int [] :=
  let value := Op.load .int (.region r.cast (.constNat 0)) .none .zero
  if addOne then .add .int .nil value (.constInt 1)
  else .mul .int .nil value (.constInt 2)

def job (M : Algebra Nat) (addOne : Bool) : Invocation Nat where
  kernel := .mk ["left", "right"] ["left", "right"] [
    .store .int [] (.region "left" (.constNat 0)) (.alg (expression addOne "left")) .none,
    .store .int [] (.region "right" (.constNat 0)) (.alg (expression addOne "right")) .none]
  backend := M
  pids := fun _ => 0

def io (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool where
  inputs := []
  outputs := [⟨.int, [], "left"⟩, ⟨.int, [], "right"⟩]
  scratch := []
  buffers := ["left", "right"]
  read := fun i => Fin.elim0 i
  inputMask := fun i => Fin.elim0 i
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  programs := fun _ => [false, true]
  invocation := fun _ event => job M event
  events := fun _ _ _ => [false, true]
  order := fun _ o _ => if o.val = 0 then [false, true] else [true, false]
  encode := id
  observe := fun _ _ => id

def vals (M : Algebra Nat) : (io M).Inputs := fun i => Fin.elim0 i
def key (M : Algebra Nat) (o : Fin 2) : (io M).OutputKey :=
  Fin.cases ⟨(0 : Fin 2), PUnit.unit⟩
    (Fin.cases ⟨(1 : Fin 2), PUnit.unit⟩ (fun i => Fin.elim0 i)) o
def outputs (M : Algebra Nat) (a b : Int) : (io M).Outputs :=
  Fin.cases (fun _ => a) (Fin.cases (fun _ => b) (fun i => Fin.elim0 i))
def raw (M : Algebra Nat) (a b : Int) : (io M).RawOutputs :=
  fun k => outputs M a b k.1 k.2

theorem unique (M : Algebra Nat) : (io M).toMemoryIO.OutputUnique (vals M) := by
  intro ⟨a, ja⟩ ⟨b, jb⟩ _ _ same
  fin_cases a <;> fin_cases b <;>
    simp_all [MemoryIO.outputAddress, io]
  all_goals
    have h : ja = jb := Subsingleton.elim (α := PUnit) _ _
    cases h
    rfl

theorem complete (M : Algebra Nat) : (io M).ScheduleComplete (vals M) := by
  refine ⟨rfl, unique M, ?_⟩
  intro ⟨o, j⟩ _
  constructor
  · fin_cases o
    · exact List.Perm.refl _
    · exact List.Perm.swap _ _ _
  · simp [io]

theorem coverage (M : Algebra Nat) (s : State Nat) : (io M).SourceCoverage (vals M) s := by
  intro k _ e listed omitted
  exact (omitted listed).elim

def final (M : Algebra Nat) (addOne : Bool) (s : State Nat) : State Nat :=
  (((job M addOne).initial s).write "left" 0
    (.mk .int (operation addOne ((s.mem "left" 0).read .int)))).write "right" 0
    (.mk .int (operation addOne ((s.mem "right" 0).read .int)))

theorem executes (M : Algebra Nat) (event : Bool) (s : State Nat) :
    (job M event).run s = some (final M event s) := by
  cases event <;>
    simp [job, Invocation.run, exec, run, step, evalExpr, expression, evalOp_unfold,
      store, TileShape.allIndices, final, operation, State.write_other,
      Region.cast, bop, numeric]

theorem safe (M : Algebra Nat) (event : Bool) (s : State Nat) (bounds : RegionName → Nat)
    (hl : 0 < bounds "left") (hr : 0 < bounds "right") : (job M event).Safe bounds s := by
  cases event <;>
    simp [job, Invocation.Safe, TraceSafe, ComputeKernel.surfaceBody,
      Safety.traceSafeList_cons, Safety.stepSafe, Safety.safeExpr, Safety.safeOp,
      Safety.safeAddress, Safety.safeMask, Safety.addressSafe, Safety.maskActive,
      expression, evalOp_unfold, Region.cast, hl, hr]

theorem final_left (M : Algebra Nat) (e : Bool) (s : State Nat) :
    (final M e s).mem "left" 0 = .mk .int (operation e ((s.mem "left" 0).read .int)) := by
  simp [final, State.write_other]

theorem final_right (M : Algebra Nat) (e : Bool) (s : State Nat) :
    (final M e s).mem "right" 0 = .mk .int (operation e ((s.mem "right" 0).read .int)) := by
  exact State.write_same _ _ _ _

theorem programs (M : Algebra Nat) (s : State Nat) : (io M).Programs (vals M) s := by
  intro current event _
  constructor
  · intro bounds backed
    exact safe M event _ bounds
      (backed.2.2.1 (0 : Fin 2) PUnit.unit trivial)
      (backed.2.2.1 (1 : Fin 2) PUnit.unit trivial)
  · refine ⟨final M event _, executes M event _, ?_⟩
    intro r off outside
    have hl := outside.1 (0 : Fin 2) PUnit.unit trivial
    have hr := outside.1 (1 : Fin 2) PUnit.unit trivial
    change r ≠ "left" ∨ off ≠ 0 at hl
    change r ≠ "right" ∨ off ≠ 0 at hr
    rw [final, State.write_other _ _ _ _ _ _ hr, State.write_other _ _ _ _ _ _ hl]
    exact congrFun (congrFun (Invocation.initial_mem _ _) r) off

theorem locality (M : Algebra Nat) (s : State Nat) : (io M).SourceLocality (vals M) s := by
  intro event _ k active a b same l r ranl ranr
  have leq := Option.some.inj ((executes M event _).symm.trans ranl)
  have req := Option.some.inj ((executes M event _).symm.trans ranr)
  subst l
  subst r
  have la := (io M).toMemoryIO.patchOutputs_read (vals M) (unique M) a s k active
  have rb := (io M).toMemoryIO.patchOutputs_read (vals M) (unique M) b s k active
  rcases k with ⟨o, j⟩
  fin_cases o
  · change (final M event _).mem "left" 0 = (final M event _).mem "left" 0
    rw [final_left, final_left]
    change ((io M).patchOutputs (vals M) a s).mem "left" 0 = _ at la
    change ((io M).patchOutputs (vals M) b s).mem "left" 0 = _ at rb
    rw [la, rb, same]
  · change (final M event _).mem "right" 0 = (final M event _).mem "right" 0
    rw [final_right, final_right]
    change ((io M).patchOutputs (vals M) a s).mem "right" 0 = _ at la
    change ((io M).patchOutputs (vals M) b s).mem "right" 0 = _ at rb
    rw [la, rb, same]

theorem source_cell_left (M : Algebra Nat) (s : State Nat) (initial : (io M).RawOutputs)
    (event : Bool) (n : Int) (t : State Nat)
    (ran : ((io M).invocation (vals M) event).run
      ((io M).currentState (vals M) s initial (key M 0) n) = some t) :
    t.mem ((io M).outputAddress (vals M) (key M 0)).1
      ((io M).outputAddress (vals M) (key M 0)).2 =
      .mk .int (operation event n) := by
  have eq := Option.some.inj ((executes M event _).symm.trans ran)
  subst t
  have bound := (io M).toMemoryIO.patchOutputs_read (vals M) (unique M)
    ((io M).toMemoryIO.updateOutput initial (key M 0) n) s (key M 0) trivial
  rw [MemoryIO.updateOutput_same] at bound
  change (final M event _).mem "left" 0 = _
  rw [final_left]
  change ((io M).currentState (vals M) s initial (key M 0) n).mem "left" 0 = _ at bound
  rw [bound]
  rfl

theorem source_cell_right (M : Algebra Nat) (s : State Nat) (initial : (io M).RawOutputs)
    (event : Bool) (n : Int) (t : State Nat)
    (ran : ((io M).invocation (vals M) event).run
      ((io M).currentState (vals M) s initial (key M 1) n) = some t) :
    t.mem ((io M).outputAddress (vals M) (key M 1)).1
      ((io M).outputAddress (vals M) (key M 1)).2 =
      .mk .int (operation event n) := by
  have eq := Option.some.inj ((executes M event _).symm.trans ran)
  subst t
  have bound := (io M).toMemoryIO.patchOutputs_read (vals M) (unique M)
    ((io M).toMemoryIO.updateOutput initial (key M 1) n) s (key M 1) trivial
  rw [MemoryIO.updateOutput_same] at bound
  change (final M event _).mem "right" 0 = _
  rw [final_right]
  change ((io M).currentState (vals M) s initial (key M 1) n).mem "right" 0 = _ at bound
  rw [bound]
  rfl

/-- Each address has its own complete order. Every successor receives the
actual previous stored value, so multiply/add and add/multiply differ. -/
theorem opposite_orders (M : Algebra Nat) (s : State Nat) :
    (io M).ScheduleComplete (vals M) ∧ (io M).Programs (vals M) s ∧
    (io M).SourceLocality (vals M) s ∧ (io M).SourceCoverage (vals M) s ∧
    (io M).SourceCellTrace (vals M) s (raw M 2 2) (key M 0) [false, true] (2 : Int) (5 : Int) ∧
    (io M).SourceCellTrace (vals M) s (raw M 2 2) (key M 1) [true, false] (2 : Int) (6 : Int) := by
  refine ⟨complete M, programs M s, locality M s, coverage M s, ?_, ?_⟩
  · apply PerAddressProgramIO.SourceCellTrace.cons_of_programs
      (io M) (vals M) s (raw M 2 2) (key M 0) false [true] (2 : Int) (4 : Int) (5 : Int)
      (programs M s) (by simp [io]) (source_cell_left M s _ false 2)
    apply PerAddressProgramIO.SourceCellTrace.cons_of_programs
      (io M) (vals M) s (raw M 2 2) (key M 0) true [] (4 : Int) (5 : Int) (5 : Int)
      (programs M s) (by simp [io]) (source_cell_left M s _ true 4)
    exact (PerAddressProgramIO.SourceCellTrace.nil_iff
      (io M) (vals M) s (raw M 2 2) (key M 0) (5 : Int) (5 : Int)).mpr rfl
  · apply PerAddressProgramIO.SourceCellTrace.cons_of_programs
      (io M) (vals M) s (raw M 2 2) (key M 1) true [false] (2 : Int) (3 : Int) (6 : Int)
      (programs M s) (by simp [io]) (source_cell_right M s _ true 2)
    apply PerAddressProgramIO.SourceCellTrace.cons_of_programs
      (io M) (vals M) s (raw M 2 2) (key M 1) false [] (3 : Int) (6 : Int) (6 : Int)
      (programs M s) (by simp [io]) (source_cell_right M s _ false 3)
    exact (PerAddressProgramIO.SourceCellTrace.nil_iff
      (io M) (vals M) s (raw M 2 2) (key M 1) (6 : Int) (6 : Int)).mpr rfl

def failingJob (M : Algebra Nat) : Invocation Nat where
  kernel := .mk [] [] [.assign .int [] "v" (.alg (.ref .int [] "missing"))]
  backend := M
  clearRegisters := true
  pids := fun _ => 0

def failingIO (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool :=
  { io M with invocation := fun _ _ => failingJob M }

/-- Actual complete source failure is rejected even when every cell trace
could otherwise be replaced by an arbitrary target. -/
theorem failing_source_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (failingIO M).Programs (vals M) s := by
  intro claimed
  obtain ⟨_, t, ran, _⟩ := claimed (raw M 2 2) false (by simp [failingIO, io])
  simp [failingIO, failingJob, Invocation.run, Invocation.initial, bindRegisters,
    exec, run, step, evalExpr, evalOp_unfold] at ran

def emptyIO (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool where
  inputs := []
  outputs := []
  scratch := []
  buffers := []
  read := fun i => Fin.elim0 i
  inputMask := fun i => Fin.elim0 i
  write := fun i => Fin.elim0 i
  outputMask := fun i => Fin.elim0 i
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  programs := fun _ => [false, true]
  invocation := fun _ event => if event then failingJob M else
    { kernel := .mk [] [] [], backend := M, pids := fun _ => 0 }
  events := fun _ i => Fin.elim0 i
  order := fun _ i => Fin.elim0 i
  encode := id
  observe := fun i => Fin.elim0 i

/-- No output lanes does not erase the second CTA's required source run. -/
theorem empty_outputs_still_require_every_cta (M : Algebra Nat) (s : State Nat) :
    ¬ (emptyIO M).Programs (fun i => Fin.elim0 i) s := by
  intro claimed
  obtain ⟨_, t, ran, _⟩ := claimed (fun k => Fin.elim0 k.1) true (by simp [emptyIO])
  simp [emptyIO, failingJob, Invocation.run, Invocation.initial, bindRegisters,
    exec, run, step, evalExpr, evalOp_unfold] at ran

def oobJob (M : Algebra Nat) : Invocation Nat where
  kernel := .mk ["left"] [] [.assign .int [] "v"
    (.alg (.load .int (.region "left" (.constNat 1)) .none .zero))]
  backend := M
  pids := fun _ => 0

def oobIO (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool :=
  { io M with invocation := fun _ _ => oobJob M }

/-- The source really loads offset 1; declaring only a scalar output window
cannot hide this active out-of-bounds access. -/
theorem out_of_bounds_source_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (oobIO M).Programs (vals M) s := by
  intro claimed
  let bounds : RegionName → Nat := fun r => if r = "left" ∨ r = "right" then 1 else 0
  have backed : (oobIO M).InBounds bounds (vals M) := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro r outside
      simp [bounds, show ¬ (r = "left" ∨ r = "right") from by simpa [oobIO, io] using outside]
    · intro i
      exact Fin.elim0 i
    · intro o j _
      fin_cases o <;> simp [oobIO, io, bounds]
    · intro k
      exact Fin.elim0 k
  have access := (claimed (raw M 2 2) false (by simp [oobIO, io])).1 bounds backed
  simp [oobIO, oobJob, Invocation.Safe, TraceSafe, ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons, Safety.stepSafe, Safety.safeExpr, Safety.safeOp,
    Safety.addressSafe, Safety.maskActive,
    evalOp_unfold, Region.cast, bounds] at access

def emptyOobIO (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool :=
  { emptyIO M with invocation := fun _ _ => oobJob M }

/-- Empty outputs do not erase the access-safety obligation either: the
actual CTA still performs its active load despite having no output lanes. -/
theorem empty_outputs_still_require_safety (M : Algebra Nat) (s : State Nat) :
    ¬ (emptyOobIO M).Programs (fun i => Fin.elim0 i) s := by
  intro claimed
  have backed : (emptyOobIO M).InBounds (fun _ => 0) (fun i => Fin.elim0 i) := by
    refine ⟨fun _ _ => rfl, ?_, ?_, ?_⟩ <;> intro i <;> exact Fin.elim0 i
  have access := (claimed (fun k => Fin.elim0 k.1) false
    (by simp [emptyOobIO, emptyIO])).1 (fun _ => 0) backed
  simp [emptyOobIO, oobJob, Invocation.Safe, TraceSafe, ComputeKernel.surfaceBody,
    Safety.traceSafeList_cons, Safety.stepSafe, Safety.safeExpr, Safety.safeOp,
    Safety.addressSafe, Safety.maskActive, evalOp_unfold, Region.cast] at access

/-- An empty/sentinel order preserves the exact selected typed value. A raw
cell with another tag cannot be declared as that initial output value. -/
theorem sentinel_exact_cell (M : Algebra Nat) (s : State Nat) (n m : Int) :
    ((io M).SourceCellTrace (vals M) s (raw M n 2) (key M 0) [] n m ↔ m = n) ∧
    ¬ (io M).OutputCells (vals M) (raw M n 2) (s.write "left" 0 (.mk .nat 7)) := by
  constructor
  · exact PerAddressProgramIO.SourceCellTrace.nil_iff _ _ _ _ _ _ _
  · intro bound
    have wrong := bound (key M 0) trivial
    change (s.write "left" 0 (.mk .nat 7)).mem "left" 0 = .mk .int n at wrong
    rw [State.write_same] at wrong
    cases wrong

def crossJob (M : Algebra Nat) : Invocation Nat where
  kernel := .mk ["right"] ["left"] [.store .int [] (.region "left" (.constNat 0))
    (.alg (.load .int (.region "right" (.constNat 0)) .none .zero)) .none]
  backend := M
  pids := fun _ => 0

def crossIO (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool :=
  { io M with invocation := fun _ _ => crossJob M }

theorem cross_runs (M : Algebra Nat) (s : State Nat) :
    (crossJob M).run s = some (((crossJob M).initial s).write "left" 0
      (.mk .int ((s.mem "right" 0).read .int))) := by
  simp [crossJob, Invocation.run, exec, run, step, evalExpr, evalOp_unfold,
    store, TileShape.allIndices, Region.cast]

/-- Equal selected input cells do not establish locality when the source
copies a different output. This dependency prevents per-address acceptance. -/
theorem cross_output_dependency_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (crossIO M).SourceLocality (vals M) s := by
  intro claimed
  have wrong := claimed false (by simp [crossIO, io]) (key M 0) trivial
    (raw M 2 3) (raw M 2 4) rfl _ _ (cross_runs M _) (cross_runs M _)
  change ((crossJob M).initial _ |>.write "left" 0 _).mem "left" 0 =
    ((crossJob M).initial _ |>.write "left" 0 _).mem "left" 0 at wrong
  rw [State.write_same, State.write_same] at wrong
  have left := (io M).toMemoryIO.patchOutputs_read (vals M) (unique M)
    (raw M 2 3) s (key M 1) trivial
  have right := (io M).toMemoryIO.patchOutputs_read (vals M) (unique M)
    (raw M 2 4) s (key M 1) trivial
  change ((io M).patchOutputs (vals M) (raw M 2 3) s).mem "right" 0 = .mk .int 3 at left
  change ((io M).patchOutputs (vals M) (raw M 2 4) s).mem "right" 0 = .mk .int 4 at right
  change Cell.mk .int (((io M).patchOutputs (vals M) (raw M 2 3) s).mem "right" 0 |>.read .int) =
    Cell.mk .int (((io M).patchOutputs (vals M) (raw M 2 4) s).mem "right" 0 |>.read .int) at wrong
  rw [left, right, Cell.read_same, Cell.read_same] at wrong
  cases wrong

def omittedIO (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool :=
  { io M with events := fun _ _ _ => [false], order := fun _ _ _ => [false] }

/-- A permutation of an incomplete declared event list does not certify the
source: the omitted add-one CTA must preserve the cell, which it does not. -/
theorem omitted_updating_cta_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (omittedIO M).SourceCoverage (vals M) s := by
  intro claimed
  have wrong := claimed (key M 0) trivial true (by simp [omittedIO, io])
    (by simp [omittedIO]) (raw M 2 2) _ (executes M true _)
  change (final M true _).mem "left" 0 = .mk .int 2 at wrong
  rw [final_left] at wrong
  have bound := (io M).toMemoryIO.patchOutputs_read (vals M) (unique M)
    (raw M 2 2) s (key M 0) trivial
  change ((io M).patchOutputs (vals M) (raw M 2 2) s).mem "left" 0 = .mk .int 2 at bound
  change Cell.mk .int (operation true
    (((io M).patchOutputs (vals M) (raw M 2 2) s).mem "left" 0 |>.read .int)) = .mk .int 2 at wrong
  rw [bound, Cell.read_same] at wrong
  simp [operation] at wrong

def duplicateIO (M : Algebra Nat) : PerAddressProgramIO Nat Nat Nat Bool :=
  { io M with
    events := (fun _ _ _ => [false, false, true])
    order := (fun _ _ _ => [false, false, true]) }

/-- Event membership alone is insufficient: duplicate occurrences are
rejected by exact occurrence accounting against the complete source list. -/
theorem duplicate_event_rejected (M : Algebra Nat) :
    ¬ (duplicateIO M).ScheduleComplete (vals M) := by
  intro claimed
  have length := (claimed.2.2 (key M 0) trivial).2.length_eq
  simp [duplicateIO, io] at length

end
end VeriTile.Triton.FP.Structural.PerAddressRegression

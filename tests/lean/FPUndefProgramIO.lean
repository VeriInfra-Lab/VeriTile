import VeriTile.Triton.Float.UndefProgramIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Triton.FP.Structural.UndefProgramRegression
noncomputable section

def job (M : Algebra Nat) (enabled : Bool) : Invocation Nat where
  kernel := .mk ["input"] ["out"] [
    .store .int [] (.region "out" (.constNat 0))
      (.alg (.load .int (.region "input" (.constNat 999))
        (.mask (.constBool enabled)) .zero)) .none]
  backend := M
  pids := fun _ => 0

def io (M : Algebra Nat) (enabled : Bool) : UndefProgramIO Nat Nat where
  inputs := [⟨.int, [], "input"⟩]
  outputs := [⟨.int, [], "out"⟩]
  scratch := []
  buffers := ["input", "out"]
  read := fun _ _ _ => 999
  inputMask := fun _ _ _ => enabled = true
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := fun u => Fin.elim0 u
  scratchMask := fun u => Fin.elim0 u
  schedule := fun _ => [job M enabled]
  encode := fun _ => 77
  observe := fun _ _ => id
  undefInputs := [⟨.int, [], "input"⟩]
  undefRead := fun _ _ _ => 999
  undefMask := fun _ _ _ => enabled = false

def inputs (M : Algebra Nat) (enabled : Bool) (memory fallback : Int) :
    (io M enabled).Inputs :=
  ⟨Fin.cases (fun _ => memory) (fun i => Fin.elim0 i),
    Fin.cases (fun _ => fallback) (fun i => Fin.elim0 i)⟩

def formula (M : Algebra Nat) (enabled : Bool) (v : (io M enabled).Inputs) :
    (io M enabled).Outputs := Fin.cases (fun _ =>
  if enabled then v.memory (0 : Fin 1) PUnit.unit else v.undefined (0 : Fin 1) PUnit.unit)
    (fun i => Fin.elim0 i)

def final (M : Algebra Nat) (enabled : Bool) (s : State Nat) : State Nat :=
  ((job M enabled).initial s).write "out" 0 (.mk .int
    (if enabled then (s.mem "input" 999).read .int else s.undef .int "input" 999))

theorem executes (M : Algebra Nat) (enabled : Bool) (s : State Nat) :
    (job M enabled).run s = some (final M enabled s) := by
  cases enabled <;>
    simp [job, Invocation.run, exec, run, step, evalExpr, evalOp_unfold,
      store, TileShape.allIndices, final, Region.cast]

theorem safe (M : Algebra Nat) (enabled : Bool) (s : State Nat)
    (bounds : RegionName → Nat) (outBacked : 0 < bounds "out")
    (inputBacked : enabled = true → 999 < bounds "input") :
    (job M enabled).Safe bounds s := by
  cases enabled <;>
    simp_all [job, Invocation.Safe, TraceSafe, ComputeKernel.surfaceBody,
      Safety.traceSafeList_cons, Safety.stepSafe, Safety.safeExpr, Safety.safeOp,
      Safety.safeAddress, Safety.safeMask, Safety.addressSafe, Safety.maskActive,
      evalOp_unfold, Region.cast]

theorem implemented (M : Algebra Nat) (enabled : Bool) :
    (io M enabled).Implements (formula M enabled) := by
  apply UndefProgramIO.ImplementsOn.intro
  · intro vals _ bounds backed initial _
    apply (scheduleSafe_singleton _ _ _).mpr
    exact safe M enabled initial bounds
      (backed.2.2.1 (0 : Fin 1) PUnit.unit trivial)
      (fun active => backed.2.1 (0 : Fin 1) PUnit.unit active)
  · intro vals _ initial binding
    refine ⟨final M enabled initial, ?_, ?_, ?_⟩
    · exact (runSchedule_singleton _ _).trans (executes M enabled initial)
    · intro o j _
      have eqo : o = (0 : Fin 1) := Fin.eq_zero o
      subst o
      have eqj : j = PUnit.unit := Subsingleton.elim (α := PUnit) _ _
      subst j
      change mapCell id ((final M enabled initial).mem "out" 0) = _
      rw [final, State.write_same]
      cases enabled
      · have h := binding.2 (0 : Fin 1) PUnit.unit rfl
        change initial.undef .int "input" 999 = vals.undefined (0 : Fin 1) PUnit.unit at h
        change Cell.mk .int (initial.undef .int "input" 999) =
          Cell.mk .int (vals.undefined (0 : Fin 1) PUnit.unit)
        exact congrArg (Cell.mk .int) h
      · have h := binding.1 (0 : Fin 1) PUnit.unit rfl
        change (initial.mem "input" 999).read .int = vals.memory (0 : Fin 1) PUnit.unit at h
        change Cell.mk .int ((initial.mem "input" 999).read .int) =
          Cell.mk .int (vals.memory (0 : Fin 1) PUnit.unit)
        exact congrArg (Cell.mk .int) h
    · intro r a outside
      have away := outside.1 (0 : Fin 1) PUnit.unit trivial
      change r ≠ "out" ∨ a ≠ 0 at away
      rw [final, State.write_other _ _ _ _ _ _ away]
      exact congrFun (congrFun (Invocation.initial_mem _ _) r) a

def fallbackState (s : State Nat) (selected unrelated : Int) : State Nat :=
  { s with undef := fun d => match d with
      | .int => fun r a => if r = "input" ∧ a = 999 then selected else unrelated
      | .real | .fp32 | .fp16 | .bf16 | .f8e4 | .f8e5 | .nat | .bool | .ptr | .blockPtr => s.undef _ }

theorem fallback_binding (M : Algebra Nat) (s : State Nat) (selected unrelated : Int) :
    (io M false).InputBinding (inputs M false 0 selected)
      (fallbackState s selected unrelated) := by
  constructor
  · intro _ _ impossible
    contradiction
  · intro u j _
    have hu : u = (0 : Fin 1) := Fin.eq_zero u
    subst u
    have hj : j = PUnit.unit := Subsingleton.elim (α := PUnit) _ _
    subst j
    change (if ("input" : RegionName) = "input" ∧ 999 = 999 then selected else unrelated) = selected
    simp

/-- The masked address needs no memory allocation, and its exact nonzero
fallback still reaches the actual stored output. Unrelated fallbacks vary freely. -/
theorem nonzero_unbacked_fallback (M : Algebra Nat) (s : State Nat) (unrelated : Int) :
    ∃ t, runSchedule ((io M false).schedule (inputs M false 0 13).memory)
      (fallbackState s 13 unrelated) = some t ∧
      t.mem "out" 0 = .mk .int 13 ∧
      (io M false).Frame (inputs M false 0 13) (fallbackState s 13 unrelated) t := by
  obtain ⟨_, t, ran, output, frame⟩ :=
    implemented M false (inputs M false 0 13) trivial _
      (fallback_binding M s 13 unrelated)
  refine ⟨t, ran, ?_, frame⟩
  have observed := output (0 : Fin 1) PUnit.unit trivial
  change mapCell id (t.mem "out" 0) = .mk .int 13 at observed
  have identity : ∀ c : Cell Nat, mapCell id c = c := by
    intro c
    cases c with
    | mk d x => cases d <;> rfl
  rw [identity] at observed
  exact observed

theorem masked_source_safe_without_input_backing (M : Algebra Nat) (s : State Nat) :
    (job M false).Safe (fun r => if r = "out" then 1 else 0) s :=
  safe M false s _ (by simp) (by intro h; contradiction)

/-- The active source read observes memory; a different fallback cannot replace it. -/
theorem active_uses_memory (M : Algebra Nat) (s : State Nat)
    (memory : (s.mem "input" 999).read .int = -7) :
    (final M true (fallbackState s 13 9999)).mem "out" 0 = .mk .int (-7) := by
  rw [final, State.write_same]
  simpa [fallbackState] using congrArg (Cell.mk .int) memory

/-- Floating encoding does not change the signed integer fallback channel. -/
theorem integer_fallback_not_encoded (M : Algebra Nat) :
    mapValue (io M false).encode .int (-13 : Int) = -13 := rfl

def failingJob (M : Algebra Nat) : Invocation Nat where
  kernel := .mk [] [] [.assign .int [] "v" (.alg (.ref .int [] "missing"))]
  backend := M
  clearRegisters := true
  pids := fun _ => 0

def failingIO (M : Algebra Nat) : UndefProgramIO Nat Nat :=
  { io M false with schedule := fun _ => [failingJob M] }

/-- An admitted logical input cannot turn actual source failure into success,
even when no memory-capacity witness is provided. -/
theorem failed_source_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (failingIO M).Implements
      (fun _ => Fin.cases (fun _ => (0 : Int)) (fun i => Fin.elim0 i)) := by
  intro claimed
  let v : (failingIO M).Inputs :=
    ⟨(inputs M false 0 13).memory, (inputs M false 0 13).undefined⟩
  have binding : (failingIO M).InputBinding v (fallbackState s 13 7) :=
    fallback_binding M s 13 7
  obtain ⟨_, t, ran, _⟩ := claimed v trivial _ binding
  simp [failingIO, io, runSchedule, failingJob, Invocation.run, Invocation.initial,
    bindRegisters, exec, run, step, evalExpr, evalOp_unfold] at ran

theorem relational_witness_bound_to_source (M : Algebra Nat) :
    (io M false).Satisfies (fun vals outputs =>
      outputs (0 : Fin 1) PUnit.unit = vals.undefined (0 : Fin 1) PUnit.unit) :=
  UndefProgramIO.SatisfiesOn.of_implementsOn (implemented M false) (fun _ _ => rfl)

#axiomsClean implemented
#axiomsClean nonzero_unbacked_fallback
#axiomsClean masked_source_safe_without_input_backing
#axiomsClean active_uses_memory
#axiomsClean integer_fallback_not_encoded
#axiomsClean failed_source_rejected
#axiomsClean relational_witness_bound_to_source
end
end VeriTile.Triton.FP.Structural.UndefProgramRegression

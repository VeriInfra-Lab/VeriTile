import VeriTile.Triton.Float.RelationalProgramIO
import Mathlib.Tactic.FinCases

namespace VeriTile.Triton.FP.Structural.RelationalOutputRegression
noncomputable section

def job (M : Algebra Nat) (fail : Bool) : Invocation Nat where
  kernel := if fail then
      .mk [] ["out"] [.assign .nat [] "value" (.alg (.ref .nat [] "missing"))]
    else .mk [] ["out"] [.store .int [] (.region "out" (.constNat 0))
      (.alg (.constInt (-7))) .none]
  backend := M
  clearRegisters := Bool.true
  pids := fun _ => 0

def io (M : Algebra Nat) (fail : Bool) : ProgramIO Nat Nat where
  inputs := [⟨.int, [], "input"⟩]
  outputs := [⟨.int, [], "out"⟩]
  scratch := []
  buffers := ["input", "out"]
  read := fun _ _ _ => 0
  inputMask := fun _ _ _ => True
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k
  schedule := fun _ => [job M fail]
  encode := id
  observe := fun _ _ => id

def outputs (M : Algebra Nat) (fail : Bool) (n : Int) : (io M fail).Outputs :=
  Fin.cases (fun _ => n) (fun k => Fin.elim0 k)

def negative (M : Algebra Nat) (fail : Bool) : (io M fail).Inputs → (io M fail).Outputs → Prop :=
  fun _ result => Int.lt (result (0 : Fin 1) PUnit.unit) (0 : Int)

/-- A non-functional logical relation is proved for all admitted signed inputs. -/
theorem source_satisfies_negative (M : Algebra Nat) :
    (io M Bool.false).Satisfies (negative M Bool.false) := by
  apply ProgramIO.Satisfies.intro
  · intro vals bounds backed s _
    have outputBound := backed.2.2.1 (0 : Fin 1) PUnit.unit trivial
    simpa [io, job, ScheduleSafe, Invocation.Safe, TraceSafe, ComputeKernel.surfaceBody,
      Safety.traceSafeList_cons, Safety.stepSafe, Safety.safeExpr, Safety.safeOp,
      Safety.safeAddress, Safety.safeMask, Safety.addressSafe, Safety.maskActive,
      evalOp_unfold] using outputBound
  · intro vals s _
    let final := ((job M Bool.false).initial s).write "out" 0 (.mk .int (-7))
    refine ⟨final, outputs M Bool.false (-7), ?_, ?_, ?_, ?_⟩
    · simp [io, job, runSchedule, Invocation.run, exec, run, step, evalExpr,
        evalOp_unfold, store, TileShape.allIndices, final]
    · intro o j _
      fin_cases o
      change mapCell id (final.mem "out" 0) = Cell.mk .int (-7)
      dsimp only [final]
      rw [State.write_same]
      rfl
    · change (-7 : Int) < 0
      decide
    · intro region offset outside
      have other := outside.1 (0 : Fin 1) PUnit.unit trivial
      change region ≠ "out" ∨ offset ≠ 0 at other
      exact State.write_other _ _ _ _ _ _ other

/-- Even a valid negative answer cannot replace the actual stored answer/tag. -/
theorem arbitrary_related_witness_rejected (M : Algebra Nat) (s : State Nat)
    (vals : (io M Bool.false).Inputs) :
    negative M Bool.false vals (outputs M Bool.false (-9)) ∧
    ¬ (io M Bool.false).OutputBinding vals (outputs M Bool.false (-9))
      (s.write "out" 0 (.mk .int (-7))) ∧
    ¬ (io M Bool.false).OutputBinding vals (outputs M Bool.false (-7))
      (s.write "out" 0 (.mk .real 7)) := by
  refine ⟨by change (-9 : Int) < 0; decide, ?_, ?_⟩
  · intro binding
    have wrong := binding (0 : Fin 1) PUnit.unit trivial
    change mapCell id ((s.write "out" 0 (.mk .int (-7))).mem "out" 0) =
      Cell.mk .int (-9) at wrong
    rw [State.write_same] at wrong
    simp [mapCell, mapValue] at wrong
  · intro binding
    have wrong := binding (0 : Fin 1) PUnit.unit trivial
    change mapCell id ((s.write "out" 0 (.mk .real 7)).mem "out" 0) =
      Cell.mk .int (-7) at wrong
    rw [State.write_same] at wrong
    simp [mapCell, mapValue] at wrong

def state : State Nat where
  mem := fun _ _ => .mk .int (-42)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- An unconstrained logical predicate does not hide source execution failure. -/
theorem failure_rejected (M : Algebra Nat) :
    ¬ (io M Bool.true).Satisfies (fun _ _ => True) := by
  intro claimed
  let vals : (io M Bool.true).Inputs :=
    Fin.cases (fun _ => (-42 : Int)) (fun k => Fin.elim0 k)
  let bounds : RegionName → Nat := fun r => if r="input" ∨ r="out" then 1 else 0
  have backed : (io M Bool.true).InBounds bounds vals := by
    refine ⟨?_, ?_, ?_, ?_⟩
    · intro r outside
      simp [bounds, show ¬ (r="input" ∨ r="out") from by simpa [io] using outside]
    · intro i j _
      fin_cases i
      simp [io, bounds]
    · intro o j _
      fin_cases o
      simp [io, bounds]
    · intro k
      exact Fin.elim0 k
  have inputs : (io M Bool.true).InputBinding vals state := by
    intro i j _
    fin_cases i
    rfl
  obtain ⟨_, final, _, ran, _⟩ := claimed vals trivial bounds backed state inputs
  simp [io, job, runSchedule, Invocation.run, Invocation.initial, bindRegisters,
    exec, run, step, evalExpr, evalOp_unfold, ComputeKernel.surfaceBody] at ran

end
end VeriTile.Triton.FP.Structural.RelationalOutputRegression

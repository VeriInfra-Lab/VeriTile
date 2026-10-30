import VeriTile.Triton.Float.ObservedProgramEquivalence
import Mathlib.Tactic.FinCases

namespace VeriTile.Triton.FP.Structural.ObservedEquivalenceRegression
noncomputable section

abbrev Symbol := Nat × Nat

def failedJob (M : Algebra Symbol) : Invocation Symbol where
  kernel := .mk [] ["out"] [.assign .nat [] "v" (.alg (.ref .nat [] "missing"))]
  backend := M
  clearRegisters := Bool.true
  pids := fun _ => 0

def io (M : Algebra Symbol) : ProgramIO Symbol Symbol where
  inputs := [⟨.fp32, [], "x"⟩]
  outputs := [⟨.fp32, [], "out"⟩]
  scratch := []
  buffers := ["x","out"]
  read := fun _ _ _ => 0
  inputMask := fun _ _ _ => True
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => [failedJob M]
  encode := id
  observe := fun _ _ => id

def candidate (M : Algebra Symbol) : ProgramIO.Variant (io M) where
  schedule := (io M).schedule
  observe := (io M).observe

def values (M : Algebra Symbol) (word provenance : Nat) : (io M).Inputs :=
  Fin.cases (fun _ => (word,provenance)) (fun i => Fin.elim0 i)

def state (word provenance : Nat) : State Symbol where
  mem := fun _ _ => .mk .fp32 (word,provenance)
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

/-- Native output observation may forget symbolic provenance without altering inputs. -/
theorem different_symbolic_outputs_observe_equal (M : Algebra Symbol) (provenance : Nat) :
    (io M).InputBinding (values M 7 provenance) (state 7 provenance) ∧
    (io M).ObservedOutputsEqual (candidate M) (fun _ _ term => term.1)
      (values M 7 provenance) (state 7 0) (state 7 42) ∧
    ¬ (io M).OutputsEqual (candidate M) (values M 7 provenance) (state 7 0) (state 7 42) := by
  constructor
  · intro i j _
    fin_cases i
    rfl
  constructor
  · intro o j _
    fin_cases o
    rfl
  · intro equal
    have impossible := equal (0 : Fin 1) PUnit.unit trivial
    change mapCell id (Cell.mk .fp32 (7,0)) = mapCell id (Cell.mk .fp32 (7,42)) at impossible
    simp [mapCell,mapValue] at impossible

/-- Equal numeric observations still cannot equate different stored dtype tags. -/
theorem different_dtype_rejected (M : Algebra Symbol) :
    ¬ (io M).ObservedOutputsEqual (candidate M) (fun _ _ term => term.1)
      (values M 7 0) (state 7 0) { state 7 0 with mem := fun _ _ => .mk .real (7,0) } := by
  intro equal
  have impossible := equal (0 : Fin 1) PUnit.unit trivial
  change mapCell (fun (x : Symbol) => x.1) (mapCell id (Cell.mk .fp32 (7,0))) =
    mapCell (fun (x : Symbol) => x.1) (mapCell id (Cell.mk .real (7,0))) at impossible
  simp [mapCell,mapValue] at impossible

/-- Even a constant observer cannot certify two source executions that fail. -/
theorem failure_not_hidden_by_constant_observation (M : Algebra Symbol) :
    ¬ (io M).ObservedEquivalent (candidate M) (fun _ _ _ => (0 : Nat)) := by
  intro equivalent
  let vals := values M 7 42
  let bounds : RegionName → Nat := fun r => if r="x" ∨ r="out" then 1 else 0
  have backed : (io M).InBounds bounds vals := by
    refine ⟨?_,?_,?_,?_⟩
    · intro r outside
      simpa [io] using (show bounds r=0 from by simp [bounds,show ¬(r="x" ∨ r="out") from by simpa [io] using outside])
    · intro i j _
      fin_cases i
      simp [bounds,io]
    · intro o j _
      fin_cases o
      simp [bounds,io]
    · intro i
      exact Fin.elim0 i
  have inputs : (io M).InputBinding vals (state 7 42) :=
    (different_symbolic_outputs_observe_equal M 42).1
  obtain ⟨_,_,left,right,ran,_⟩ := equivalent vals trivial bounds backed (state 7 42) inputs
  simp [io,failedJob,runSchedule,Invocation.run,Invocation.initial,bindRegisters,
    exec,run,step,evalExpr,evalOp_unfold,ComputeKernel.surfaceBody] at ran

end
end VeriTile.Triton.FP.Structural.ObservedEquivalenceRegression

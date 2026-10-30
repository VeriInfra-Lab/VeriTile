import VeriTile.Triton.Float.ProgramEquivalence
import VeriTile.Triton.Float.ReadProgramIO
import Mathlib.Tactic.FinCases

/-! Public IO semantics: successful execution, typed defaults and exact exterior frames. -/

namespace VeriTile.Triton.FP.Structural.InterfaceRegression
noncomputable section
set_option linter.unusedSimpArgs false

def failingJob (M : Algebra Nat) : Invocation Nat where
  kernel := .mk [] [] [.assign .nat [] "v" (.alg (.ref .nat [] "missing"))]
  backend := M
  clearRegisters := true
  pids := fun _ => 0

def failingIO (M : Algebra Nat) : ProgramIO Nat Nat where
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
  schedule := fun _ => [failingJob M]
  encode := id
  observe := fun i => Fin.elim0 i

def sameVariant (M : Algebra Nat) : ProgramIO.Variant (failingIO M) where
  schedule := (failingIO M).schedule
  observe := (failingIO M).observe

/-- Even identical, memory-safe sources fail equivalence if they cannot execute. -/
theorem identical_failure_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (failingIO M).Equivalent (sameVariant M) := by
  intro h
  let vals : (failingIO M).Inputs := fun i => Fin.elim0 i
  have hb : (failingIO M).InBounds (fun _ => 0) vals := by
    refine ⟨fun _ _ => rfl, ?_, ?_, ?_⟩ <;> intro i <;> exact Fin.elim0 i
  have hi : (failingIO M).InputBinding vals s := by intro i; exact Fin.elim0 i
  obtain ⟨_, _, left, right, executed, _⟩ := h vals trivial (fun _ => 0) hb s hi
  simp [failingIO, failingJob, runSchedule, Invocation.run, Invocation.initial,
    bindRegisters, exec, run, step, evalExpr, evalOp_unfold] at executed

def vetoedIO (M : Algebra Nat) : ReadProgramIO Nat Nat where
  inputs := [⟨.real, [], "x"⟩]
  outputs := [⟨.real, [], "x"⟩]
  scratch := []
  buffers := ["x"]
  read := fun _ _ _ => 0
  inputMask := fun _ _ _ => True
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => False
  scratchWrite := fun i => Fin.elim0 i
  scratchMask := fun i => Fin.elim0 i
  schedule := fun _ => [{ kernel := .mk ["x"] ["x"] [], backend := M, pids := fun _ => 0 }]
  encode := id
  observe := fun _ _ => id
  resultRead := fun _ _ _ => 0
  resultMask := fun _ _ _ => True

def mismatched : State Nat where
  mem := fun _ _ => Cell.mk .nat 7
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

def zeroInput (M : Algebra Nat) : (vetoedIO M).Inputs
  | ⟨0, _⟩ => fun _ => by change Nat; exact 0
  | ⟨i+1,h⟩ => False.elim (by simp [vetoedIO] at h)

def zeroOutput (M : Algebra Nat) : (vetoedIO M).Outputs
  | ⟨0, _⟩ => fun _ => by change Nat; exact 0
  | ⟨i+1,h⟩ => False.elim (by simp [vetoedIO] at h)

/-- Existing typed-read default semantics admit the unchanged non-float cell. -/
theorem mismatch_readback_preserved (M : Algebra Nat) :
    (vetoedIO M).InputBinding (zeroInput M) mismatched ∧
    (vetoedIO M).OutputReadback (zeroInput M) (zeroOutput M) mismatched := by
  constructor
  · intro i j _
    fin_cases i
    rfl
  · intro i j _
    fin_cases i
    rfl

/-- The observed result lane remains in the raw frame because the store is vetoed. -/
theorem result_lane_is_still_framed (M : Algebra Nat) (before after : State Nat)
    (frame : (vetoedIO M).Frame (zeroInput M) before after) :
    after.mem "x" 0 = before.mem "x" 0 := by
  apply frame "x" 0
  constructor
  · intro i j active
    exact False.elim active
  · intro i
    exact Fin.elim0 i

/-- The read-view result is not silently replaced by a stronger stored-tag result. -/
theorem exact_tag_claim_rejected :
    ¬ mapCell id (mismatched.mem "x" 0) = Cell.mk (α := Nat) .real 0 := by
  simp [mismatched, mapCell, mapValue]

end
end VeriTile.Triton.FP.Structural.InterfaceRegression

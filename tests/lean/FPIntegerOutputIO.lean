import VeriTile.Triton.Float.IntegerOutputIO
import Mathlib.Tactic.FinCases

namespace VeriTile.Triton.FP.Structural.IntegerOutputRegression
open QuantizeIntegerStage
noncomputable section

/-- Signed and unsigned views retain distinct meanings for the same low byte. -/
theorem signed_unsigned_low8_distinct :
    (IntegerOutputView.signedWidth 8).apply 255 = -1 ∧
    (IntegerOutputView.unsignedWidth 8).apply (-1) = 255 ∧
    (IntegerOutputView.identity).apply 257 = 257 := by
  decide

def memory : MemoryIO Nat where
  inputs := []
  outputs := [⟨.int, [], "out"⟩]
  scratch := []
  buffers := ["out"]
  read := fun i => Fin.elim0 i
  inputMask := fun i => Fin.elim0 i
  write := fun _ _ _ => 0
  outputMask := fun _ _ _ => True
  scratchWrite := fun k => Fin.elim0 k
  scratchMask := fun k => Fin.elim0 k

def vals : memory.Inputs := fun i => Fin.elim0 i
def output (n : Int) : memory.Outputs :=
  Fin.cases (fun _ => n) (fun i => Fin.elim0 i)

/-- Observation sees byte 1; the source's actual raw integer cell stays 257. -/
theorem low8_observation_preserves_raw_cell (s : State Nat) :
    memory.IntegerOutputBinding (fun _ _ => id) (fun _ _ => .unsignedWidth 8)
      vals (output 1) (s.write "out" 0 (.mk .int 257)) ∧
    (s.write "out" 0 (.mk .int 257)).mem "out" 0 = Cell.mk .int 257 := by
  constructor
  · intro o j _
    fin_cases o
    change (IntegerOutputView.unsignedWidth 8).mapCell
      (mapCell id ((s.write "out" 0 (.mk .int 257)).mem "out" 0)) = Cell.mk .int 1
    rw [State.write_same]
    rfl
  · exact State.write_same _ _ _ _

/-- A numerically equal natural cell cannot masquerade as an integer output. -/
theorem wrong_dtype_rejected (s : State Nat) :
    ¬ memory.IntegerOutputBinding (fun _ _ => id) (fun _ _ => .unsignedWidth 8)
      vals (output 255) (s.write "out" 0 (.mk .nat 255)) := by
  intro bound
  have wrong := bound (0 : Fin 1) PUnit.unit trivial
  change (IntegerOutputView.unsignedWidth 8).mapCell
    (mapCell id ((s.write "out" 0 (.mk .nat 255)).mem "out" 0)) = Cell.mk .int 255 at wrong
  rw [State.write_same] at wrong
  simp [IntegerOutputView.mapCell, IntegerOutputView.mapValue, mapCell, mapValue] at wrong

def failingJob (M : Algebra Nat) : IntegerInvocation Nat where
  kernel := .mk [] [] [.assign .int [] "output" (.llrint (.alg (.ref .real [] "x")))]
  backend := M
  integerStorage := fun _ => none
  runtimeRegisters := outputRegister
  round := none
  pids := fun _ => 0

def failingIO (M : Algebra Nat) : IntegerProgramIO Nat Nat where
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

def sameVariant (M : Algebra Nat) : IntegerProgramIO.Variant (failingIO M) where
  schedule := (failingIO M).schedule
  observe := (failingIO M).observe

/-- Equality of observations never certifies identical failed source runs. -/
theorem equal_failures_rejected (M : Algebra Nat) (s : State Nat) :
    ¬ (failingIO M).IntegerObservedEquivalent (sameVariant M) (fun i => Fin.elim0 i) := by
  intro claimed
  let inputs : (failingIO M).Inputs := fun i => Fin.elim0 i
  have backed : (failingIO M).InBounds (fun _ => 0) inputs := by
    refine ⟨fun _ _ => rfl, ?_, ?_, ?_⟩ <;> intro i <;> exact Fin.elim0 i
  have bound : (failingIO M).InputBinding inputs s := by
    intro i
    exact Fin.elim0 i
  obtain ⟨_, _, left, right, executed, _⟩ :=
    claimed inputs trivial (fun _ => 0) backed s bound
  simp [failingIO, failingJob, runIntegerSchedule, IntegerInvocation.run,
    IntegerInvocation.typedKernel, QuantizeIntegerStage.exec, QuantizeIntegerStage.run,
    QuantizeIntegerStage.step, QuantizeIntegerStage.evalExpr, ComputeKernel.surfaceBody] at executed

end
end VeriTile.Triton.FP.Structural.IntegerOutputRegression

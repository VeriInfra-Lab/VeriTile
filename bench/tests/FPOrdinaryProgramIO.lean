import VeriTile.Triton.Float.OrdinaryProgramIO
import VeriTile.Meta.StatementAudit
import Mathlib.Tactic

namespace VeriTile.Triton.FP.Structural.OrdinaryRegression
noncomputable section
set_option maxHeartbeats 1500000
set_option maxRecDepth 10000
variable {α : Type} [Inhabited α]
open scoped Classical

def producer (A : Algebra α) : Invocation α where
  kernel := .mk [] [] [
    .store .int [] (.region "shared" (.constNat 0)) (.alg (.constInt 9)) .none]
  backend := A
  pids := fun _ => 0

def consumer (A : Algebra α) : Invocation α where
  kernel := .mk [] [] [
    .assign .int [] "v" (.alg (.load .int (.region "shared" (.constNat 0)) .none .zero)),
    .store .int [] (.region "out" (.constNat 0)) (.alg (.ref .int [] "v")) .none]
  backend := A
  pids := fun _ => 0

abbrev jobs (A : Algebra α) := [producer A,consumer A]
def writes (i : Fin 2) : RegionName → Nat → Prop :=
  if i.val = 0 then fun r off => r="shared" ∧ off=0
  else fun r off => r="out" ∧ off=0

def producerResult (A : Algebra α) (s : State α) : State α :=
  (producer A).initial s |>.write "shared" 0 (.mk .int 9)
def consumerResult (A : Algebra α) (s : State α) : State α :=
  ((consumer A).initial s).setReg "v" .int [] (fun _ => (s.mem "shared" 0).read .int)
    |>.write "out" 0 (.mk .int ((s.mem "shared" 0).read .int))
def rows (A : Algebra α) (s : State α) (i : Fin 2) : State α :=
  if i.val = 0 then producerResult A s else consumerResult A s

theorem producer_run (A : Algebra α) (s : State α) :
    (producer A).run s=some (producerResult A s) := by
  simp [Invocation.run,producer,producerResult,exec,run,step,
    evalExpr,evalOp_unfold,store,TileShape.allIndices]

theorem consumer_run (A : Algebra α) (s : State α) :
    (consumer A).run s=some (consumerResult A s) := by
  simp [Invocation.run,consumer,consumerResult,exec,run,step,
    evalExpr,evalOp_unfold,store,TileShape.allIndices,
    State.setReg,Invocation.initial,bindRegisters]

def gridRun (A : Algebra α) (s : State α) :
    OrdinaryRun (jobs A) writes s (OrdinaryRun.merge writes s (rows A s)) := by
  apply OrdinaryRun.of_rows
  · intro i
    fin_cases i
    · exact producer_run A s
    · exact consumer_run A s
  · intro i r off miss
    fin_cases i
    · change (if r="shared" ∧ off=0 then Cell.mk .int (9 : Int) else s.mem r off) = s.mem r off
      exact if_neg (by simpa [writes] using miss)
    · change (if r="out" ∧ off=0 then Cell.mk .int ((s.mem "shared" 0).read .int)
          else s.mem r off) = s.mem r off
      exact if_neg (by simpa [writes] using miss)
  · intro i j r off hi hj
    fin_cases i <;> fin_cases j <;> simp_all [writes]

/-- An ordinary grid's second CTA reads the original input, even though a
separate CTA writes that address. This distinguishes it from a serial launch. -/
theorem reads_common_initial (A : Algebra α) (s : State α) :
    (OrdinaryRun.merge writes s (rows A s)).mem "out" 0=
      .mk .int ((s.mem "shared" 0).read .int) := by
  rw [(gridRun A s).readback 1 "out" 0 (by simp [writes])]
  rfl

/-- Serial execution would instead read the producer's nine. This is why an
ordinary grid cannot be silently replaced by a source launch sequence. -/
theorem serial_reads_written (A : Algebra α) (s : State α) :
    ∃ final, runSchedule (jobs A) s = some final ∧
      final.mem "out" 0 = Cell.mk .int (9 : Int) := by
  refine ⟨consumerResult A (producerResult A s), ?_, ?_⟩
  · simp only [jobs, runSchedule, producer_run, Option.bind_some, consumer_run]
  · rfl

/-- Exact cells outside both source write sets remain unchanged. -/
theorem exterior (A : Algebra α) (s : State α) (r : RegionName) (off : Nat)
    (outside : (r≠"shared" ∨ off≠0) ∧ (r≠"out" ∨ off≠0)) :
    (OrdinaryRun.merge writes s (rows A s)).mem r off=s.mem r off := by
  apply (gridRun A s).frame
  intro i
  fin_cases i
  · change ¬ (r="shared" ∧ off=0)
    tauto
  · change ¬ (r="out" ∧ off=0)
    tauto

theorem empty_grid_identity (s t : State α)
    (h : OrdinaryRun [] (fun i=>Fin.elim0 i) s t) : t.mem=s.mem :=
  OrdinaryRun.nil_memory _ s t h

#axiomsClean producer_run
#axiomsClean consumer_run
#axiomsClean gridRun
#axiomsClean reads_common_initial
#axiomsClean serial_reads_written
#axiomsClean exterior
#axiomsClean empty_grid_identity
end
end VeriTile.Triton.FP.Structural.OrdinaryRegression

import VeriTile.Triton.Concurrent.EventBoundary
import VeriTile.Triton.Concurrent.EventFrame

/-! Focused semantic regressions for actual current-memory operations and
constructive controller completion. These require no arithmetic assumptions. -/
namespace VeriTile.Triton.Concurrent.Regression

/-- Atomic addition sees the value written by the preceding source event. -/
theorem store_then_atomicAdd (b : Backend V) (s : State V) (region : Nat) (address : Int) :
    (run b [.store region address (.input 0 0),
      .atomicAdd region address (.input 1 0)] s).mem region address=
      b.add (s.input 0 0) (s.input 1 0) := by
  simp [run,execute,Expr.eval,update]

/-- The branch after CAS tests the returned old word, not the replacement. -/
def casController : Controller where
  threads := 1
  enabled := .constant true
  entry := 0
  initialLocals := []
  code := [
    .emit (.exchange 0 (.literal 0) (.literal 7)) (.goto 1 []),
    .emit (.compareExchange 0 (.literal 0) (.literal 7) (.literal 9) 0)
      (.branch (.returnedEq 0 (.literal 7)) (.goto 2 []) (.goto 3 []))]
  terminal := [2]

theorem missing_instruction_fails (b : Backend V) (s : ControlState V) (thread : Nat)
    (missing : casController.code[s.pc thread]?=none) :
    casController.step b thread s=none := by
  simp [Controller.step,missing]

/-- Merely starting a valid controller does not certify completion. -/
theorem empty_schedule_not_done (s : State V) : ¬casController.Done (casController.initial s) := by
  intro done
  have atZero := done 0 (by decide) (by rfl)
  simp [casController,Controller.initial] at atZero

private def casFinal (b : Backend V) (s : State V) : ControlState V :=
  ⟨execute b (.compareExchange 0 0 7 9 0 0)
      (execute b (.exchange 0 0 7) (Program.initialValues b s)),
    fun thread=>if thread=0 then 2 else 0,fun _ _=>0⟩

theorem cas_completed_run (b : Backend V) (s : State V) :
    casController.run b [0,0] (casController.initial (Program.initialValues b s))=
      some ([.exchange 0 0 7,.compareExchange 0 0 7 9 0 0],casFinal b s) := by
  simp [Controller.run,Controller.step,Controller.initial,casController,casFinal,
    Code.decode,EventCode.instantiate,Next.eval,Test.eval,NumExpr.eval,AddrExpr.eval,
    execute,update,Program.initialValues]
  constructor
  · funext thread
    by_cases atZero : thread=0 <;> simp [update,atZero]
  · funext thread slot
    simp [update]

theorem cas_completed_values (b : Backend V) (s : State V) :
    casController.Done (casFinal b s) ∧
    (casFinal b s).values.returned 0 0=7 ∧ (casFinal b s).values.words 0 0=9 := by
  refine ⟨?_,?_,?_⟩
  · intro thread inside enabled
    have atZero : thread=0 := by simpa [casController] using inside
    simp [casController,casFinal,atZero]
  · simp [casFinal,execute,update]
  · simp [casFinal,execute,update]

/-- Combining the finite run and terminal fact yields actual completion. -/
theorem cas_program_runs (b : Backend V) (s : State V) :
    Program.Runs b (.controller casController) s
      [.exchange 0 0 7,.compareExchange 0 0 7 9 0 0] (casFinal b s).values :=
  ⟨[0,0],casFinal b s,cas_completed_run b s,(cas_completed_values b s).1,rfl⟩

#axiomsClean store_then_atomicAdd
#axiomsClean missing_instruction_fails
#axiomsClean empty_schedule_not_done
#axiomsClean cas_completed_run
#axiomsClean cas_completed_values
#axiomsClean cas_program_runs
end VeriTile.Triton.Concurrent.Regression

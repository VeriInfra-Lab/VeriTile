import VeriTile.Triton.Float.BackendMemory
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.BackendMemory
open VeriTile.Triton FP.Structural FP.BackendMemory
noncomputable section
variable {α : Type} [Inhabited α] (M : Algebra α)

def failing : Program M := ⟨.mk [] [] [
  .assign .nat [] "result" (.alg (.ref .nat [] "absent"))]⟩

/-- Identical failure is deliberately insufficient for successful equivalence. -/
theorem identical_failures_rejected (s : State α) :
    ¬Equivalent (failing M) (failing M) := by
  apply not_equivalent_of_failure (s := {s with regs := fun _ _ _ => none})
  simp [failing,exec,run,step,evalExpr,evalOp_unfold]

def localLeft : Program M := ⟨.mk [] [] [
  .assign .nat [] "left" (.alg (.constNat 1))]⟩
def localRight : Program M := ⟨.mk [] [] [
  .assign .nat [] "right" (.alg (.constNat 2))]⟩

/-- Unrelated private register names/values do not prevent exact memory equality. -/
theorem different_registers_same_memory : Equivalent (localLeft M) (localRight M) := by
  intro s
  refine ⟨s.setReg "left" .nat [] (fun _ => 1),s.setReg "right" .nat [] (fun _ => 2),?_,?_,rfl⟩
  all_goals simp [localLeft,localRight,exec,run,step,evalExpr,evalOp_unfold]

#axiomsClean identical_failures_rejected
#axiomsClean different_registers_same_memory
end
end VeriTile.Tests.BackendMemory

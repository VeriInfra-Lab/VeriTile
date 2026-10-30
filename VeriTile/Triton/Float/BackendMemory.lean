import VeriTile.Triton.Float.Structural
import VeriTile.Spec

/-! Successful whole-memory equivalence under an explicit instruction backend.

The backend is fixed in the program type, so both actual source kernels use
the same instruction interpretation. No input-domain restriction, executor
callback, memory observer, or safety assumption is installed by this interface.
Access safety is a separate IO obligation. This relation also covers arbitrary
initial states for which no finite-memory IO application is being claimed.
-/
namespace VeriTile.Triton.FP.BackendMemory
open Structural
noncomputable section

structure Program {α : Type} (backend : Algebra α) where
  kernel : ComputeKernel

/-- Both complete source executions succeed and agree on every typed memory
cell. Private registers and program metadata are not memory observations. -/
def Equivalent {α : Type} [Inhabited α] {backend : Algebra α}
    (lhs rhs : Program backend) : Prop :=
  ∀ s, ∃ a b, exec backend lhs.kernel s = some a ∧
    exec backend rhs.kernel s = some b ∧ a.mem = b.mem

instance programSyntax {α : Type} [Inhabited α] {backend : Algebra α} :
    Spec.ProgramSyntax (Program backend) where
  Statement := ComputeStmt
  Signature := List RegionName × List RegionName
  signature p := (p.kernel.inputs,p.kernel.outputs)
  body p := p.kernel.surfaceBody
  structural := some Equivalent
  sameContext := fun lhs rhs => lhs=rhs

/-- A failed source execution cannot satisfy the successful-memory relation,
even when both programs fail in exactly the same way. -/
theorem not_equivalent_of_failure {α : Type} [Inhabited α] {backend : Algebra α}
    (lhs rhs : Program backend) (s : State α) (failed : exec backend lhs.kernel s=none) :
    ¬Equivalent lhs rhs := by
  intro equivalent
  obtain ⟨a,b,ran,_,_⟩ := equivalent s
  rw [failed] at ran
  cases ran

end
end VeriTile.Triton.FP.BackendMemory

import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import VeriTile.Triton.Float.GuardedIO

/-! Guarded numerical equivalence for concrete finite launch programs.
This adapter uses ProgramIO's existing execution, safety, output and frame
judgment. The only implementation data are actual invocations and storage
precision tags; it introduces neither an interpreter nor a numerical rule. -/
namespace VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

structure GuardedProgram {α : Type} (io : ProgramIO α α)
    (allowed : io.Inputs → Prop) (domain : FP.Guarded.Domain α) where
  jobs : List (Invocation α)
  outputPrecision : List TileDType

namespace GuardedProgram
variable {α : Type} [Inhabited α] {io : ProgramIO α α}
  {allowed : io.Inputs → Prop} {domain : FP.Guarded.Domain α}

def implementation (p : GuardedProgram io allowed domain) :
    Implementation io allowed .exactCells where
  schedule := fun _ => p.jobs
  observe := fun _ _ => id

def Equivalent (p q : GuardedProgram io allowed domain) : Prop :=
  Implementation.Equivalent p.implementation q.implementation

def Models (R : Spec.Assumptions FP.GuardedFragment)
    (p : GuardedProgram io allowed domain) : Prop :=
  ∀ job ∈ p.jobs, FP.Guarded.Models R job.backend domain

/-- Each numerical step supplies both model certificates and the complete IO
proof. These must not be implications: transitivity may choose a different
intermediate backend, including one that does not model the rule set. -/
def Numerical (R : Spec.Assumptions FP.GuardedFragment)
    (p q : GuardedProgram io allowed domain) : Prop :=
  p.Models R ∧ q.Models R ∧ Equivalent p q

instance : Spec.ProgramSyntax (GuardedProgram io allowed domain) where
  Statement := FP.GuardedFragment
  theory := FP.Guarded.fp32Theory
  Signature := List TileDType
  signature := GuardedProgram.outputPrecision
  body := fun p => p.jobs.map fun job => ⟨[], job.kernel.surfaceBody, .real, none⟩
  structural := some Equivalent
  numerical := Numerical
  sameContext := Eq

/-- A numerical step still has to prove the complete, existing IO judgment.
Model certificates alone never discharge an implementation comparison. -/
theorem of_equivalent (p q : GuardedProgram io allowed domain)
    (precision : p.outputPrecision = q.outputPrecision)
    (proof : Equivalent p q) (R : Spec.Assumptions FP.GuardedFragment) :
    Spec.Equiv p q R :=
  Spec.FloatingPoint.ofStructural rfl precision proof

/-- Numeric laws may be used in `proof` through the supplied certificates;
neither a missing model nor a model certificate replaces execution evidence. -/
theorem of_numerical (p q : GuardedProgram io allowed domain)
    (precision : p.outputPrecision = q.outputPrecision)
    (R : Spec.Assumptions FP.GuardedFragment)
    (leftModel : p.Models R) (rightModel : q.Models R)
    (proof : Equivalent p q) : Spec.Equiv p q R :=
  Spec.FloatingPoint.ofNumerical rfl precision ⟨leftModel, rightModel, proof⟩

#axiomsClean of_equivalent
#axiomsClean of_numerical

end GuardedProgram
end
end VeriTile.Triton.FP.Structural.ProgramIO

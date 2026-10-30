import VeriTile.Spec
import VeriTile.Triton.Float.InputBindingIO
import VeriTile.Meta.StatementAudit

/-! Syntax for the existing complete source-schedule equivalence judgments.

The public IO, logical domain and input-binding mode are fixed by the type.
Each implementation supplies only actual structural invocations and its output
observer. There is no custom runner or correctness callback. A syntax statement
retains the entire implementation data, so syntax-only rules cannot erase an
input-dependent schedule. Structural steps reuse the existing successful-run,
access-safety, output-cell and exterior-frame obligations.
-/
namespace VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

inductive InputBindingMode where
  | decoded
  | exactCells

/-- Another concrete implementation of fixed public IO on the same domain. -/
structure Implementation {α β : Type} (io : ProgramIO α β)
    (domain : io.Inputs → Prop) (mode : InputBindingMode := .decoded) where
  schedule : io.Inputs → List (Invocation β)
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α

namespace Implementation
variable {α β : Type} {io : ProgramIO α β} {domain : io.Inputs → Prop}
  {mode : InputBindingMode}

def toVariant (p : Implementation io domain mode) : Variant io where
  schedule := p.schedule
  observe := p.observe

def toProgramIO (p : Implementation io domain mode) : ProgramIO α β :=
  p.toVariant.toProgramIO

def original (io : ProgramIO α β) (domain : io.Inputs → Prop)
    (mode : InputBindingMode := .decoded) : Implementation io domain mode where
  schedule := io.schedule
  observe := io.observe

def ofVariant (candidate : Variant io) (domain : io.Inputs → Prop)
    (mode : InputBindingMode := .decoded) : Implementation io domain mode where
  schedule := candidate.schedule
  observe := candidate.observe

/-- Rebase only the implementation on the left IO; all public data stays fixed. -/
def relativeVariant (left right : Implementation io domain mode) : Variant left.toProgramIO where
  schedule := right.schedule
  observe := right.observe

def Equivalent [Inhabited β] (left right : Implementation io domain mode) : Prop :=
  match mode with
  | .decoded => left.toProgramIO.EquivalentOn (left.relativeVariant right) domain
  | .exactCells => left.toProgramIO.EquivalentOnCells (left.relativeVariant right) domain

/-- A singleton statement retains the actual schedule and observer as data.
The empty statement theory permits no numeric or schedule rewrite by itself. -/
instance programSyntax [Inhabited β] : Spec.ProgramSyntax (Implementation io domain mode) where
  Statement := Implementation io domain mode
  Signature := Unit
  signature := fun _ => ()
  body := fun p => [p]
  structural := some Equivalent
  sameContext := Eq

theorem of_equivalentOn [Inhabited β] (candidate : Variant io)
    (proof : io.EquivalentOn candidate domain) :
    Spec.Equiv (original io domain) (ofVariant candidate domain)
      ([] : Spec.Assumptions (Implementation io domain)) := by
  exact Spec.FloatingPoint.ofStructural rfl rfl proof

theorem of_equivalent [Inhabited β] (candidate : Variant io)
    (proof : io.Equivalent candidate) :
    Spec.Equiv (original io (fun _ => True)) (ofVariant candidate (fun _ => True))
      ([] : Spec.Assumptions (Implementation io (fun _ => True))) := by
  exact of_equivalentOn candidate proof

theorem of_equivalentOnCells [Inhabited β] (candidate : Variant io)
    (proof : io.EquivalentOnCells candidate domain) :
    Spec.Equiv (original io domain .exactCells) (ofVariant candidate domain .exactCells)
      ([] : Spec.Assumptions (Implementation io domain .exactCells)) := by
  exact Spec.FloatingPoint.ofStructural rfl rfl proof

theorem of_equivalentCells [Inhabited β] (candidate : Variant io)
    (proof : io.EquivalentCells candidate) :
    Spec.Equiv (original io (fun _ => True) .exactCells)
      (ofVariant candidate (fun _ => True) .exactCells)
      ([] : Spec.Assumptions (Implementation io (fun _ => True) .exactCells)) := by
  exact of_equivalentOnCells candidate proof

#axiomsClean of_equivalentOn
#axiomsClean of_equivalentOnCells
#guard_msgs (drop info) in
#auditModuleAxioms
end Implementation
end
end VeriTile.Triton.FP.Structural.ProgramIO

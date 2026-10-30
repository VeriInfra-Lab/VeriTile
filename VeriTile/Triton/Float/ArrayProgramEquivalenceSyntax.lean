import VeriTile.Spec
import VeriTile.Triton.Float.ArrayProgramEquivalence
import VeriTile.Meta.StatementAudit

/-! Two concrete source schedules over the existing named-array interface.
The public arrays, finite read footprint, logical domain and observations are
retained. Implementations contain actual invocations, never custom executors
or correctness callbacks. Structural equivalence requires both successful safe
executions, all observed output cells and both exterior memory frames. -/
namespace VeriTile.Triton.FP.Structural.ArrayProgramIO
noncomputable section

structure Implementation {α β : Type} (io : ArrayProgramIO α β)
    (domain : io.Inputs → Prop) where
  schedule : io.Inputs → List (Invocation β)
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → α

namespace Implementation
variable {α β : Type} {io : ArrayProgramIO α β} {domain : io.Inputs → Prop}

def toVariant (p : Implementation io domain) : Variant io where
  schedule := p.schedule
  observe := p.observe

def toArrayProgramIO (p : Implementation io domain) : ArrayProgramIO α β :=
  p.toVariant.toArrayProgramIO

def original (io : ArrayProgramIO α β) (domain : io.Inputs → Prop) :
    Implementation io domain where
  schedule := io.schedule
  observe := io.observe

def ofVariant (candidate : Variant io) (domain : io.Inputs → Prop) :
    Implementation io domain where
  schedule := candidate.schedule
  observe := candidate.observe

def relativeVariant (left right : Implementation io domain) : Variant left.toArrayProgramIO where
  schedule := right.schedule
  observe := right.observe

def Equivalent [Inhabited β] (left right : Implementation io domain) : Prop :=
  left.toArrayProgramIO.EquivalentOn (left.relativeVariant right) domain

/-- The complete implementation is retained as statement data; an empty
experimental rule set cannot erase a schedule or supply its execution proof. -/
instance programSyntax [Inhabited β] : Spec.ProgramSyntax (Implementation io domain) where
  Statement := Implementation io domain
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

#axiomsClean of_equivalentOn
#guard_msgs (drop info) in
#auditModuleAxioms
end Implementation
end
end VeriTile.Triton.FP.Structural.ArrayProgramIO

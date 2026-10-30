import VeriTile.Triton.Float.IntegerOutputIO
import VeriTile.Meta.Specification

/-! Reader syntax for two actual integer-stage launch schedules.

The memory layout, input encoding and independent input domain are fixed in
the program type. Each implementation stores only its concrete invocations
and floating payload observation. Its syntax body retains those data; syntax
steps also require equality of the implementations. A structural step must
prove the existing complete two-schedule judgment, including both successful
runs, source access safety, typed output cells and both raw exterior frames.
-/
namespace VeriTile.Triton.FP.Structural.IntegerProgramIO
noncomputable section
open scoped VeriTile.Spec

structure Implementation {α β : Type} (io : IntegerProgramIO α β)
    (domain : io.Inputs → Prop) where
  schedule : io.Inputs → List (IntegerInvocation β)
  observe : (o : Fin io.outputs.length) → TileIndex io.outputs[o].shape → β → α

namespace Implementation
variable {α β : Type} {io : IntegerProgramIO α β} {domain : io.Inputs → Prop}

def toIO (program : Implementation io domain) : IntegerProgramIO α β :=
  { io with schedule := program.schedule, observe := program.observe }

def original (io : IntegerProgramIO α β) (domain : io.Inputs → Prop) :
    Implementation io domain := ⟨io.schedule, io.observe⟩

def fromVariant (domain : io.Inputs → Prop) (candidate : Variant io) :
    Implementation io domain := ⟨candidate.schedule, candidate.observe⟩

def asVariant (lhs rhs : Implementation io domain) : Variant lhs.toIO :=
  ⟨rhs.schedule, rhs.observe⟩

def Equivalent [Inhabited β] (lhs rhs : Implementation io domain) : Prop :=
  lhs.toIO.EquivalentOn (asVariant lhs rhs) domain

instance [Inhabited β] : Spec.ProgramSyntax (Implementation io domain) where
  Statement := Implementation io domain
  Signature := PUnit
  signature := fun _ => ()
  body := fun program => [program]
  structural := some Equivalent
  sameContext := Eq

def assumptions (io : IntegerProgramIO α β) (domain : io.Inputs → Prop) :
    Spec.Assumptions (Implementation io domain) := []

theorem equiv_of_equivalentOn [Inhabited β] (candidate : Variant io)
    (proof : io.EquivalentOn candidate domain) :
    original io domain ≡[assumptions io domain] fromVariant domain candidate := by
  exact Spec.FloatingPoint.ofStructural rfl rfl proof

theorem equiv_of_equivalent [Inhabited β] (candidate : Variant io)
    (proof : io.Equivalent candidate) :
    original io (fun _ => True) ≡[assumptions io (fun _ => True)]
      fromVariant (fun _ => True) candidate :=
  equiv_of_equivalentOn candidate proof

#axiomsClean equiv_of_equivalentOn
#axiomsClean equiv_of_equivalent
end Implementation

/-! The closed integer output views retain their exact existing semantics.
The selected view is fixed in the program type and never supplied by a proof
as an arbitrary integer function. The underlying execution and raw frames
remain the integer-stage ones above. -/
structure ObservedImplementation {α β : Type} (io : IntegerProgramIO α β)
    (domain : io.Inputs → Prop)
    (views : (o : Fin io.outputs.length) → TileIndex io.outputs[o].shape → IntegerOutputView)
    extends Implementation io domain

namespace ObservedImplementation
variable {α β : Type} {io : IntegerProgramIO α β} {domain : io.Inputs → Prop}
    {views : (o : Fin io.outputs.length) → TileIndex io.outputs[o].shape → IntegerOutputView}

def original (io : IntegerProgramIO α β) (domain : io.Inputs → Prop)
    (views : (o : Fin io.outputs.length) → TileIndex io.outputs[o].shape → IntegerOutputView) :
    ObservedImplementation io domain views := ⟨Implementation.original io domain⟩

def fromVariant (domain : io.Inputs → Prop)
    (views : (o : Fin io.outputs.length) → TileIndex io.outputs[o].shape → IntegerOutputView)
    (candidate : Variant io) : ObservedImplementation io domain views :=
  ⟨Implementation.fromVariant domain candidate⟩

def Equivalent [Inhabited β] (lhs rhs : ObservedImplementation io domain views) : Prop :=
  lhs.toImplementation.toIO.IntegerObservedEquivalentOn
    (Implementation.asVariant lhs.toImplementation rhs.toImplementation) views domain

instance [Inhabited β] : Spec.ProgramSyntax (ObservedImplementation io domain views) where
  Statement := ObservedImplementation io domain views
  Signature := PUnit
  signature := fun _ => ()
  body := fun program => [program]
  structural := some Equivalent
  sameContext := Eq

def assumptions (io : IntegerProgramIO α β) (domain : io.Inputs → Prop)
    (views : (o : Fin io.outputs.length) → TileIndex io.outputs[o].shape → IntegerOutputView) :
    Spec.Assumptions (ObservedImplementation io domain views) := []

theorem equiv_of_equivalentOn [Inhabited β] (candidate : Variant io)
    (proof : io.IntegerObservedEquivalentOn candidate views domain) :
    original io domain views ≡[assumptions io domain views]
      fromVariant domain views candidate := by
  exact Spec.FloatingPoint.ofStructural rfl rfl proof

theorem equiv_of_equivalent [Inhabited β] (candidate : Variant io)
    (proof : io.IntegerObservedEquivalent candidate views) :
    original io (fun _ => True) views ≡[assumptions io (fun _ => True) views]
      fromVariant (fun _ => True) views candidate :=
  equiv_of_equivalentOn candidate proof

#axiomsClean equiv_of_equivalentOn
#axiomsClean equiv_of_equivalent
end ObservedImplementation
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Triton.FP.Structural.IntegerProgramIO

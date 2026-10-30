import VeriTile.Spec
import VeriTile.Triton.Float.PerAddressProgramEquivalence

/-! Actual per-address program-pair syntax on fixed public IO and domain.
The complete source invocation and observer are data. The structural judgment
retains all source traces, each independent address order and both frames.
No numeric, schedule or whole-output rewrite is built into this view. -/
namespace VeriTile.Triton.FP.Structural.PerAddressProgramIO
noncomputable section

structure Implementation {α β γ Event : Type} (io : PerAddressProgramIO α β γ Event)
    (domain : io.Inputs → Prop) where
  invocation : io.Inputs → Event → Invocation β
  observe : (o : Fin io.outputs.length) → TileIndex (io.outputs[o]).shape → β → γ

namespace Implementation
variable {α β γ Event : Type} {io : PerAddressProgramIO α β γ Event} {domain : io.Inputs → Prop}

def toVariant (p : Implementation io domain) : Variant io := ⟨p.invocation,p.observe⟩
def toPerAddressIO (p : Implementation io domain) : PerAddressProgramIO α β γ Event := p.toVariant.toPerAddressIO

def original (io : PerAddressProgramIO α β γ Event) (domain : io.Inputs → Prop) : Implementation io domain :=
  ⟨io.invocation,io.observe⟩
def ofVariant (candidate : Variant io) (domain : io.Inputs → Prop) : Implementation io domain :=
  ⟨candidate.invocation,candidate.observe⟩

def relativeVariant (left right : Implementation io domain) : Variant left.toPerAddressIO :=
  ⟨right.invocation,right.observe⟩

def Equivalent [Inhabited β] (left right : Implementation io domain) : Prop :=
  left.toPerAddressIO.EquivalentOnCells (left.relativeVariant right) domain

instance programSyntax [Inhabited β] : Spec.ProgramSyntax (Implementation io domain) where
  Statement := Implementation io domain
  Signature := Unit
  signature := fun _ => ()
  body := fun p => [p]
  structural := some Equivalent
  sameContext := Eq

theorem of_equivalentOnCells [Inhabited β] (candidate : Variant io)
    (proof : io.EquivalentOnCells candidate domain) :
    Spec.Equiv (original io domain) (ofVariant candidate domain)
      ([] : Spec.Assumptions (Implementation io domain)) := by
  exact Spec.FloatingPoint.ofStructural rfl rfl proof

#axiomsClean of_equivalentOnCells
end Implementation
end
end VeriTile.Triton.FP.Structural.PerAddressProgramIO

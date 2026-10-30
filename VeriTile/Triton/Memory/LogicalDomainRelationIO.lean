import VeriTile.Triton.Memory.LogicalIO

namespace VeriTile.Triton

/-- Preserve a public logical input domain together with a relational output.
The type index is the existing full judgment; neither it nor its executor is
supplied as mutable IO data. Both coordinate maps are bijections. -/
structure LogicalDomainRelationIO (RawInputs RawOutputs Inputs Outputs : Type)
    (judgment : (RawInputs → Prop) → (RawInputs → RawOutputs → Prop) → Prop) where
  inputs : RawInputs ≃ Inputs
  outputs : RawOutputs ≃ Outputs

namespace LogicalDomainRelationIO
variable {RI RO I O : Type} {judgment : (RI → Prop) → (RI → RO → Prop) → Prop}

def SatisfiesOn (io : LogicalDomainRelationIO RI RO I O judgment)
    (domain : I → Prop) (relation : I → O → Prop) : Prop :=
  judgment (fun raw => domain (io.inputs raw))
    (fun raw result => relation (io.inputs raw) (io.outputs result))

scoped notation:25 io " ⊨ᵣ{" domain "} " relation =>
  LogicalDomainRelationIO.SatisfiesOn io domain relation

end LogicalDomainRelationIO
end VeriTile.Triton

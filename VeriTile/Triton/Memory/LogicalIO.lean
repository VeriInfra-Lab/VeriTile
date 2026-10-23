import Mathlib.Logic.Equiv.Defs

/-! Bijective logical coordinates for an existing, fixed IO judgment.

The judgment is a type parameter, never an IO field, executor or proof oracle.
Specialized views must instantiate it with their complete existing judgment,
including its observer, initialization, cell domain and schedule. These adapters
only transport input/output coordinates. Task-level connection proofs identify
the transported formula with the previous independent mathematical target.
-/
namespace VeriTile.Triton

/-- Coordinate data only. Both directions retain every raw input and output. -/
structure LogicalIO (RawInputs RawOutputs Inputs Outputs : Type)
    (judgment : (RawInputs → RawOutputs) → Prop) where
  inputs : RawInputs ≃ Inputs
  outputs : RawOutputs ≃ Outputs

namespace LogicalIO
variable {RI RO I O : Type} {judgment : (RI → RO) → Prop}

def lift (io : LogicalIO RI RO I O judgment) (f : I → O) : RI → RO :=
  fun raw => io.outputs.symm (f (io.inputs raw))

def Implements (io : LogicalIO RI RO I O judgment) (f : I → O) : Prop :=
  judgment (io.lift f)

scoped infix:25 " ⊨ " => LogicalIO.Implements

theorem implements_iff (io : LogicalIO RI RO I O judgment) (f : I → O) :
    (io ⊨ f) ↔ judgment (io.lift f) := Iff.rfl
end LogicalIO

/-- Logical input domains remain public and are pulled back bijectively. -/
structure LogicalDomainIO (RawInputs RawOutputs Inputs Outputs : Type)
    (judgment : (RawInputs → Prop) → (RawInputs → RawOutputs) → Prop) where
  inputs : RawInputs ≃ Inputs
  outputs : RawOutputs ≃ Outputs

namespace LogicalDomainIO
variable {RI RO I O : Type} {judgment : (RI → Prop) → (RI → RO) → Prop}

def lift (io : LogicalDomainIO RI RO I O judgment) (f : I → O) : RI → RO :=
  fun raw => io.outputs.symm (f (io.inputs raw))

def ImplementsOn (io : LogicalDomainIO RI RO I O judgment)
    (domain : I → Prop) (f : I → O) : Prop :=
  judgment (fun raw => domain (io.inputs raw)) (io.lift f)

scoped notation:25 io " ⊨{" domain "} " f => LogicalDomainIO.ImplementsOn io domain f

theorem implementsOn_iff (io : LogicalDomainIO RI RO I O judgment)
    (domain : I → Prop) (f : I → O) :
    (io ⊨{domain} f) ↔ judgment (fun raw => domain (io.inputs raw)) (io.lift f) := Iff.rfl
end LogicalDomainIO

/-- A relational observation is transported without replacing it by a function
or losing typed/multiple outputs. The original judgment remains fixed. -/
structure LogicalRelationIO (RawInputs RawOutputs Inputs Outputs : Type)
    (judgment : (RawInputs → RawOutputs → Prop) → Prop) where
  inputs : RawInputs ≃ Inputs
  outputs : RawOutputs ≃ Outputs

namespace LogicalRelationIO
variable {RI RO I O : Type} {judgment : (RI → RO → Prop) → Prop}

def Satisfies (io : LogicalRelationIO RI RO I O judgment) (relation : I → O → Prop) : Prop :=
  judgment (fun raw result => relation (io.inputs raw) (io.outputs result))

scoped notation:25 io " ⊨ᵣ " relation => LogicalRelationIO.Satisfies io relation
end LogicalRelationIO
end VeriTile.Triton

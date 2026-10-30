import VeriTile.Triton.Float.GuardedRules
import VeriTile.Triton.Float.StructuralIO
import VeriTile.Triton.Float.ExecutionProfile

/-! Shared observations and domains for guarded execution. These definitions
do not assume any numerical laws or select an implementation model. -/
namespace VeriTile.Triton.FP.Guarded
open Structural

abbrev Domain (α : Type) := GuardKind → α → Prop

def ScalarDomain {α : Type} (D : Domain α) (guards : List OperandGuard)
    (s : State α) : Prop :=
  ∀ g ∈ guards, ∃ v, s.regs .real [] g.operand = some v ∧ D g.kind (v PUnit.unit)

def TypedScalarDomain {α : Type} (D : Domain α) (dtype : FloatDType)
    (guards : List OperandGuard) (s : State α) : Prop :=
  ∀ g ∈ guards, ∃ v, s.regs dtype.toTileDType [] g.operand = some v ∧
    D g.kind (toFloat dtype (v PUnit.unit))

@[simp] theorem typedScalarDomain_real {α : Type} (D : Domain α)
    (guards : List OperandGuard) (s : State α) :
    TypedScalarDomain D .real guards s = ScalarDomain D guards s := rfl

def scalarOutput {α : Type} (dtype : FloatDType) (s : State α) : Option (TileIndex [] → α) :=
  (s.regs dtype.toTileDType [] "out").map (fun v i => toFloat dtype (v i))

@[simp] theorem scalarOutput_real {α : Type} (s : State α) :
    scalarOutput .real s = s.regs .real [] "out" := by
  simp [scalarOutput, toFloat]

def fragmentAlgebra {α : Type} (f : GuardedFragment) (M : Algebra α) : Algebra α :=
  match f.defaultPrecision with
  | none => M
  | some p => M.withDefaultPrecision p

@[simp] theorem fragmentAlgebra_none {α : Type} (f : GuardedFragment) (M : Algebra α)
    (h : f.defaultPrecision = none) : fragmentAlgebra f M = M := by
  simp [fragmentAlgebra, h]

end VeriTile.Triton.FP.Guarded

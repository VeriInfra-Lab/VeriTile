import VeriTile.Triton.Float.ProfiledRewrite

/-! Whole-kernel equivalence for FP32-profiled sources with different local
registers. Both executions must succeed and agree on every typed memory cell;
the kernel ports and actual rewrite-site domain remain part of the signature.
This relation is not a contextual statement rewrite. -/
namespace VeriTile.Triton.FP.ProfiledMemory
open Structural Guarded ProfiledRewrite

/-- Memory observations may need a scalar rule only on selected lanes.
The client must still prove that other lanes cannot affect observed memory. -/
inductive Site where
  | unmasked (site : ProfiledRewrite.Site)
  | maskedScalarFP32 (before : List ComputeStmt) (shape : TileShape)
      (mask : Op .bool shape) (tile : Op .real shape) (scalar : Op .real [])
  | maskedBinaryFP32 (before : List ComputeStmt) (shape : TileShape)
      (mask : Op .bool shape) (left right : Op .real shape)

def Site.Holds {α : Type} [Inhabited α] (M : Algebra α) (D : Domain α)
    (s : State α) : Site → Prop
  | .unmasked site => site.Holds M D s
  | .maskedScalarFP32 before _ mask tile scalar =>
    ∀ t, run (engine M) before s = some t →
      ∀ active a b, Structural.evalOp (engine M) none mask t = some active →
        Structural.evalOp (engine M) (some .fp32) tile t = some a →
        Structural.evalOp (engine M) (some .fp32) scalar t = some b →
        ∀ i, active i = true → D .finite (a i) ∧ D .finite (b PUnit.unit)
  | .maskedBinaryFP32 before _ mask left right =>
    ∀ t, run (engine M) before s = some t →
      ∀ active a b, Structural.evalOp (engine M) none mask t = some active →
        Structural.evalOp (engine M) (some .fp32) left t = some a →
        Structural.evalOp (engine M) (some .fp32) right t = some b →
        ∀ i, active i = true → D .finite (a i) ∧ D .finite (b i)

structure Program where
  kernel : ComputeKernel
  domain : List Site

def Equivalent (R : Spec.Assumptions GuardedFragment) (lhs rhs : Program) : Prop :=
  ∀ (α : Type) [Inhabited α] (M : Algebra α) (D : Domain α), Models R M D →
    ∀ s, (∀ site ∈ lhs.domain, site.Holds M D s) →
      ∃ a b, Structural.exec (engine M) lhs.kernel s = some a ∧
        Structural.exec (engine M) rhs.kernel s = some b ∧
        a.mem = b.mem

instance : Spec.ProgramSyntax Program where
  Statement := GuardedFragment
  theory := Guarded.fp32Theory
  Signature := (List RegionName × List RegionName) × List Site
  signature p := ((p.kernel.inputs, p.kernel.outputs), p.domain)
  body p := [⟨[], p.kernel.surfaceBody, .real, none⟩]
  structural := some (fun _ _ => False)
  numerical := Equivalent
  sameContext := fun lhs rhs => lhs = rhs

end VeriTile.Triton.FP.ProfiledMemory

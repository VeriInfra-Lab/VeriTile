import VeriTile.Triton.Float.WhereRewrite
namespace VeriTile.Tests.FPWhereContext
open VeriTile.Triton VeriTile.Triton.FP VeriTile.Triton.FP.ProfiledRewrite
/-- The two migrated sites read a register and a scalar literal. Their finite
operand domains are definitionally unchanged by the explicit fp32 annotation. -/
theorem same_guard {α : Type} [Inhabited α] (M : Structural.Algebra α)
    (D : Guarded.Domain α) (s : Structural.State α) (before : List ComputeStmt)
    (n : Nat) (rest : TileShape) (name : RegName) (scale : ℝ) :
    (Site.mulScalar before n rest (.ref .real (n :: rest) name) (.const scale)).Holds M D s ↔
    (Site.mulScalarFP32 before n rest (.ref .real (n :: rest) name) (.const scale)).Holds M D s := Iff.rfl
#axiomsClean same_guard
end VeriTile.Tests.FPWhereContext

import Mathlib.Analysis.SpecialFunctions.Pow.Real
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Target
open scoped BigOperators
noncomputable section
/-- Independent initial-input GEMV with source-order optional output addition. -/
def dot (K : Nat) (x w : Nat → ℝ) : ℝ := ∑ k : Fin K,x k.val*w k.val
def output (K : Nat) (x w : Nat → ℝ) (old : ℝ) (add active : Bool) : ℝ :=
  if active then if add then dot K x w+old else dot K x w else old
end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Target

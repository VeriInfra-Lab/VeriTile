import bench.repairs.tritonbench_g.reversed_cumsum.IO
import bench.repairs.tritonbench_g.reversed_cumsum.Proofs.IOComplete

namespace VeriTile.Bench.Repairs.ReversedCumsum.ReaderFormat
open VeriTile.Triton Interface
open scoped BigOperators VeriTile.Triton.LogicalIO
noncomputable section

/-- The new time/feature formula transports to exactly the old suffix target. -/
theorem formula_eq (h : Call) (c : Config) (program : Geometry.Program h) :
    (suffixIO h c program).lift (fun xs time feature =>
      ∑ t : Fin h.T, if time ≤ t then xs t feature else 0) = reference h := by
  rfl

/-- Equivalence of the complete old and new judgments, not only output values. -/
theorem statement_iff (h : Call) (c : Config) (program : Geometry.Program h) :
    (suffixIO h c program ⊨ fun xs time feature =>
      ∑ t : Fin h.T, if time ≤ t then xs t feature else 0) ↔
      (io h c program).Implements (reference h) := by
  change (io h c program).Implements _ ↔ _
  rw [formula_eq]

theorem correct (h : Call) (distinct : h.input ≠ h.output)
    (c : Config) (chosen : c ∈ configurations) (program : Geometry.Program h) :
    suffixIO h c program ⊨ fun xs time feature =>
      ∑ t : Fin h.T, if time ≤ t then xs t feature else 0 :=
  (statement_iff h c program).mpr (IOComplete.implements h distinct c chosen program)

end
end VeriTile.Bench.Repairs.ReversedCumsum.ReaderFormat

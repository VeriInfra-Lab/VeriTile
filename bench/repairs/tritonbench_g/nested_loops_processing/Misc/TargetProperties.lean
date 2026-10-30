import bench.repairs.tritonbench_g.nested_loops_processing.Correct

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- Compatibility for the retained flat mathematical target. The public
`Correct.integerTarget` now uses logical matrix coordinates. -/
theorem target_real (rows cols i : Nat) :
    (ReaderBefore.integerTarget rows cols i : ℝ) = realTarget rows cols i :=
  Formula.integer_real rows cols i

/-- The current logical matrix target has the same exact Real interpretation. -/
theorem logical_target_real (rows cols : Nat) (positive : 0<rows*cols)
    (row : Fin rows) (column : Fin cols) :
    (integerTarget rows cols row column : ℝ) = realTarget rows cols (row.val*cols+column.val) := by
  change (LogicalReader.integerTarget rows cols row column : ℝ) = _
  rw [LogicalReader.formula_eq rows cols positive row column]
  exact target_real rows cols (row.val*cols+column.val)

#axiomsClean target_real
#axiomsClean logical_target_real
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct

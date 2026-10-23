import bench.repairs.tritonbench_g.nested_loops_processing.Correct

namespace VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct
open VeriTile.Triton FP.Structural
noncomputable section

/-- The retained mathematical Real target is exactly the integer result's cast. -/
theorem target_real (rows cols i : Nat) :
    (integerTarget rows cols i : ℝ) = realTarget rows cols i := Formula.integer_real rows cols i

#axiomsClean target_real
end
end VeriTile.Bench.Repairs.NestedLoopsProcessing.Correct

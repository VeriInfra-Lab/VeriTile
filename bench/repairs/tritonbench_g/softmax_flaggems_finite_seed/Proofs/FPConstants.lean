import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPLoop
import Mathlib.Data.Rat.Cast.OfScientific

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPConstants
/-- Exact denotation of the source decimal literal, before native encoding. -/
theorem decimal_zero : (0.0 : ℝ) = 0 := by
  norm_num [OfScientific.ofScientific, Rat.ofScientific]
  rfl
#axiomsClean decimal_zero
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.FPConstants

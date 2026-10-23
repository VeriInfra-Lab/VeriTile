import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.FiniteResults
import bench.tritonbench_g.spinning_lock_reduction.Proofs.Real

namespace VeriTile.Bench.Repairs.SpinningLock.Correct
open VeriTile VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
noncomputable section

/-- The producer's finite output satisfies the next stage's input domain. -/
theorem producer_feeds_consumer (h : HostArgs) (input : Formula.Values (ℝ) h count)
    (admitted : Formula.Domain (0 : ℝ) h .producers input) :
    Formula.Domain (0 : ℝ) h .consumers (Formula.stage (0 : ℝ) h .producers input) :=
  FiniteResults.producer_to_consumer_domain (0 : ℝ) h input admitted

#axiomsClean producer_feeds_consumer
end
end VeriTile.Bench.Repairs.SpinningLock.Correct

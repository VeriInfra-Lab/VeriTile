import bench.repairs.tritonbench_g.spinning_lock_reduction.Proofs.FiniteResults
import bench.tritonbench_g.spinning_lock_reduction.Proofs.FP

namespace VeriTile.Bench.Repairs.SpinningLock.FPEquiv
open VeriTile VeriTile.Triton VeriTile.Bench.TritonBenchG.SpinningLock
noncomputable section

/-- The producer's finite output satisfies the next stage's input domain. -/
theorem producer_feeds_consumer (h : HostArgs) (input : Formula.Values (FP.Value .fp32) h count)
    (admitted : Formula.Domain fp32.zero h .producers input) :
    Formula.Domain fp32.zero h .consumers (Formula.stage fp32.zero h .producers input) :=
  FiniteResults.producer_to_consumer_domain fp32.zero h input admitted

#axiomsClean producer_feeds_consumer
end
end VeriTile.Bench.Repairs.SpinningLock.FPEquiv

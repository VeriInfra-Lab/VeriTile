import bench.repairs.tritonbench_g.fused_recurrent_retention.Proofs.HostPlans
import bench.tritonbench_g.fused_recurrent_retention.Proofs.CorrectPublic
import bench.tritonbench_g.fused_recurrent_retention.Proofs.FPEquivPublic
namespace VeriTile.Bench.Repairs.FusedRecurrentRetention
noncomputable section
namespace Retained
open VeriTile.Bench.TritonBenchG.FusedRecurrentRetention.Full
/-- The unchanged full forward source proof remains available, including both
feature flags and its finite allocation/frame proof. It is not represented as
native numerical conformance. -/
theorem forward (p : Layout) (r : ForwardRegions) (seed final : Bool) :
    Correctness.ForwardImplements (ForwardSource.kernel p r seed final Bool.false) p r seed final :=
  Correctness.forward_implements p r seed final Bool.false

/-- Historical backward complete-source proof: intentionally retained with its
original zero final-gradient semantics. The repaired F contribution is proved
by reverseCarry_real/adjoint_real, not by relabelling this old theorem. -/
theorem backward_zero_final (p : Layout) (r : BackwardRegions) (seed : Bool) :
    Correctness.BackwardImplements (BackwardSource.kernel p r seed Bool.false) p r seed :=
  Correctness.backward_implements p r seed Bool.false

#axiomsClean forward
#axiomsClean backward_zero_final
#axiomsClean Equivalence.fused_recurrent_retention_forward_typed_success
#axiomsClean Equivalence.fused_recurrent_retention_backward_typed_success
end Retained
end
end VeriTile.Bench.Repairs.FusedRecurrentRetention

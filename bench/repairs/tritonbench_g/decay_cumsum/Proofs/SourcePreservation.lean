import bench.repairs.tritonbench_g.decay_cumsum.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DecayCumsum.SourcePreservation
/-- All complete forward/preparation/backward original functions are retained. -/
theorem source : Kernels.source = LegacySurface.source := by
  funext kind
  cases kind <;> rfl

/-- All three complete existing candidates are retained. -/
theorem indexCommuted : Kernels.indexCommuted = LegacySurface.indexCommuted := by
  funext kind
  cases kind <;> rfl

#axiomsClean source
#axiomsClean indexCommuted
end VeriTile.Bench.Repairs.DecayCumsum.SourcePreservation

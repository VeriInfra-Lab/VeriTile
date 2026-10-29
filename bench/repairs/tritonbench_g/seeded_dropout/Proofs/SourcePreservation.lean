import bench.repairs.tritonbench_g.seeded_dropout.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.SeededDropout.SourcePreservation
/-- Complete original source equality for all seed carriers and parameters. -/
theorem kernel : SeededDropout.kernel = LegacySurface.kernel := by
  funext kind
  cases kind <;> rfl

/-- Complete candidate source equality for all seed carriers and parameters. -/
theorem indexCommutedKernel : SeededDropout.indexCommutedKernel = LegacySurface.indexCommutedKernel := by
  funext kind
  cases kind <;> rfl

#axiomsClean kernel
#axiomsClean indexCommutedKernel
end VeriTile.Bench.Repairs.SeededDropout.SourcePreservation

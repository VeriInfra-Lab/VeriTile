import bench.repairs.tritonbench_g.uniform_sampling.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.UniformSampling.SourcePreservation
/-- Complete original source equality, including every scalar word and launch parameter. -/
theorem kernel : UniformSampling.kernel = LegacySurface.kernel := by rfl

/-- Complete existing candidate source equality, including all four Philox fields and stores. -/
theorem indexCommutedKernel : UniformSampling.indexCommutedKernel = LegacySurface.indexCommutedKernel := by rfl

#axiomsClean kernel
#axiomsClean indexCommutedKernel
end VeriTile.Bench.Repairs.UniformSampling.SourcePreservation

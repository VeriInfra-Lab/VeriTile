import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource.SourcePreservation
/-- Both complete selected source functions are retained for every Parameters value. -/
theorem kernel : KernelSource.kernel = LegacySurface.kernel := by
  funext version
  cases version <;> rfl

#axiomsClean kernel
end VeriTile.Bench.Repairs.QuantTransposeKernel.KernelSource.SourcePreservation

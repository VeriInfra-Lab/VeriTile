import bench.repairs.tritonbench_g.isfinite_kernel.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.IsfiniteKernel.SourcePreservation
/-- Both complete precision branches of the original strided source are retained. -/
theorem source : IsfiniteKernel.source = LegacySurface.source := by
  funext precision
  cases precision <;> rfl

/-- Both complete precision branches of the existing index candidate are retained. -/
theorem indexCommuted : IsfiniteKernel.indexCommuted = LegacySurface.indexCommuted := by
  funext precision
  cases precision <;> rfl

#axiomsClean source
#axiomsClean indexCommuted
end VeriTile.Bench.Repairs.IsfiniteKernel.SourcePreservation

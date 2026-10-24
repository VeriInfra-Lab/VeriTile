import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.MaskedAddCuda.SourcePreservation
/-- All mask-width branches and every original source parameter are retained. -/
theorem kernel : MaskedAddCuda.kernel = LegacySurface.kernel := by
  funext storage
  cases storage <;> rfl

/-- The complete existing addition-order candidate is retained. -/
theorem commutedKernel : MaskedAddCuda.commutedKernel = LegacySurface.commutedKernel := by
  funext storage
  cases storage <;> rfl

#axiomsClean kernel
#axiomsClean commutedKernel
end VeriTile.Bench.Repairs.MaskedAddCuda.SourcePreservation

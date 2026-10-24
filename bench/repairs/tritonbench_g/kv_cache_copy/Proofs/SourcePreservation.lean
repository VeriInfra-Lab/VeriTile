import bench.repairs.tritonbench_g.kv_cache_copy.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.KvCacheCopy.Source.SourcePreservation
/-- Equality of both complete source functions, for every parameter value. -/
theorem kernel : Source.kernel=LegacySurface.kernel := by
  funext commuted
  cases commuted <;> rfl
#axiomsClean kernel
end VeriTile.Bench.Repairs.KvCacheCopy.Source.SourcePreservation

import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Proofs.LegacySurface
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.SourcePreservation
/-- All prior storage choices and original/dotLeft source interfaces are
unchanged by replacing the former whole-source macro with ordinary quotations. -/
theorem source (qk value hist dhist : TypedSource.Storage) (swapped : Bool) :
    TypedSource.specialize qk value hist dhist swapped=
      LegacySurface.specialize qk value hist dhist swapped := by
  cases hist <;> cases dhist <;> cases swapped <;> rfl
#axiomsClean source
end VeriTile.Bench.Repairs.ChunkBwdDqkg.SourcePreservation

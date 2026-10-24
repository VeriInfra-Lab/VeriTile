import bench.repairs.tritonbench_g.bmm_chunk_bwd.Proofs.LegacySurface
namespace VeriTile.Bench.Repairs.BmmChunkBwd.SourcePreservation
/-- Complete original/candidate interfaces and executed ASTs are unchanged. -/
theorem bindMain (version : Bool) : Kernels.bindMain version=LegacySurface.bindMain version := by
  cases version <;> rfl
#axiomsClean bindMain
end VeriTile.Bench.Repairs.BmmChunkBwd.SourcePreservation

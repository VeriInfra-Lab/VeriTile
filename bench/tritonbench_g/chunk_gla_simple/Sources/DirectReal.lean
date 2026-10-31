import bench.tritonbench_g.chunk_gla_simple.Sources.Execution
import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_HostReal

namespace VeriTile.Bench.Repairs.ChunkGlaSimple.DirectReal
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ChunkGlaSimple Math Real Host
noncomputable section
set_option maxHeartbeats 8000000

def footprint (c : Shape) : WriteFootprint := fun cell => cell.1="o" ∧ cell.2<c.heads*c.T*c.V

end
end VeriTile.Bench.Repairs.ChunkGlaSimple.DirectReal

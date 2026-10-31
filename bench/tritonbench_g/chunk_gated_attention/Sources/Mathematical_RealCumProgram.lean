import bench.tritonbench_g.chunk_cumsum_vector.Sources.Mathematical_RealInit

/-! Shared mathematical layout types for the source-derived cumulative proof. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumProgram
open VeriTile.Triton
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

abbrev Layout := ChunkCumsumVector.RealProgram.Layout
abbrev Launch := ChunkCumsumVector.RealProgram.Launch

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.RealCumProgram

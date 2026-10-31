import bench.tritonbench_g.chunk_gla_simple.Kernels
import bench.tritonbench_g.chunk_gla_simple.Sources.Mathematical_WholeShape

/-! Transparent sharing of the current repair quotation. The full FP32 AST
was checked equal to the former source family before retiring that copy. -/
namespace VeriTile.Bench.Repairs.ChunkGlaSimple
open VeriTile.Triton
abbrev Storage := VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Storage
abbrev Shape := VeriTile.Bench.TritonBenchG.ChunkGlaSimple.Host.Shape
abbrev source := VeriTile.Bench.Repairs.ChunkGlaSimpleMinDotTile.fp32Original
end VeriTile.Bench.Repairs.ChunkGlaSimple

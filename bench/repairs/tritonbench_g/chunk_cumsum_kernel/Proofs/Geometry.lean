import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.chunk_cumsum_kernel.Kernels
import bench.tritonbench_g.chunk_cumsum_kernel.Kernels

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel
open VeriTile.Bench.TritonBenchG.ChunkCumsumKernel
abbrev kernel := TypedSource.original

theorem Config.positive (c : Config) : 0<c.block := by cases c <;> decide
theorem configs_complete (c : Config) : c∈configs := by cases c <;> simp [configs]
end VeriTile.Bench.Repairs.ChunkCumsumKernel

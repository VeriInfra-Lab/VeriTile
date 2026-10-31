import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.chunk_cumsum_kernel.Kernels

namespace VeriTile.Bench.Repairs.ChunkCumsumKernel

theorem Config.positive (c : Config) : 0<c.block := by cases c <;> decide
theorem configs_complete (c : Config) : c∈configs := by cases c <;> simp [configs]
end VeriTile.Bench.Repairs.ChunkCumsumKernel

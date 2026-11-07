import bench.tritonbench_g.chunk_linear_attn.Kernels

namespace VeriTile.Bench.Repairs.ChunkLinearAttn
open VeriTile.Triton
/-- The complete Python candidate hoists the common Q/K head address. -/
def forwardOHoisted (p : Parameters) (scale : ℝ) : ComputeKernel :=
  forwardOHoistedKernel p scale
end VeriTile.Bench.Repairs.ChunkLinearAttn

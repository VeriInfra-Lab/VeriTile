import bench.tritonbench_g.parallel_retention_attention.Kernels
import bench.tritonbench_g.parallel_attention.Proofs.Sources.Mathematical_FPBackwardGraph

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward
open VeriTile.Triton FP.Structural
noncomputable section
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)

def shellSource (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
  backwardPastedKernel c.q c.k c.v c.do_ dq c.dk c.dv
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.B c.H c.T c.scale
    c.BTL c.BTS c.BK c.BV c.K c.V

def fusedShellSource (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
  fusedBackwardPastedKernel c.q c.k c.v c.do_ dq c.dk c.dv
    c.s_qk_h c.s_qk_t c.s_qk_d c.s_vo_h c.s_vo_t c.s_vo_d c.B c.H c.T c.scale
    c.BTL c.BTS c.BK c.BV c.K c.V
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPBackward

import bench.tritonbench_g.parallel_retention_attention.Kernels
import bench.tritonbench_g.parallel_attention.Proofs.Sources.Mathematical_FPBackwardGraph

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel
open VeriTile.Triton FP.Structural
noncomputable section
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (BackwardConfig)
namespace FPStages
def fusedForwardSource
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  fusedForwardPastedKernel q k v o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T scale BTL BTS BK BV K V
end FPStages
namespace FPQuery
def fusedSource (c : BackwardConfig) (dq : RegionName) : ComputeKernel :=
  instantiate_triton_procedure fusedQueryProcedureSource procedure_args [
    i_bh := c.i_bh, i_c := c.i_c, i_k := c.i_k, i_v := c.i_v, i_h := c.i_h, k := c.k, v := c.v, «do» := c.do_, dq := dq, s_qk_h := c.s_qk_h, s_qk_t := c.s_qk_t, s_qk_d := c.s_qk_d, s_vo_h := c.s_vo_h, s_vo_t := c.s_vo_t, s_vo_d := c.s_vo_d, B := c.B, H := c.H, T := c.T, scale := c.scale, BTL := c.BTL, BTS := c.BTS, BK := c.BK, BV := c.BV, DK := c.K, DV := c.V
  ]
end FPQuery
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel

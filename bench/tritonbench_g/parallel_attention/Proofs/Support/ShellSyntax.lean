import bench.tritonbench_g.parallel_attention.Kernels

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def shellSource
    (q k v do_ dz dq dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  backwardPastedKernel q k v do_ dz dq dk dv s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d scale B H T K V BTL BTS BK BV

def fusedShellSource
    (q k v do_ dz dq dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  fusedBackwardKernel q k v do_ dz dq dk dv s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d scale B H T K V BTL BTS BK BV

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel

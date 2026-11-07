import bench.tritonbench_g.parallel_attention.Kernels

namespace VeriTile.Bench.Repairs.ParallelAttentionKernel
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000

def querySource
    (q k v do_ dz dq : RegionName)
    (i_bh i_c i_k i_v : Nat)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  instantiate_triton_procedure queryProcedureSource procedure_args [i_bh := i_bh, i_c := i_c, i_k := i_k, i_v := i_v, i_h := 0, q := q, k := k, v := v, «do» := do_, dz := dz, dq := dq, s_k_h := s_k_h, s_k_t := s_k_t, s_k_d := s_k_d, s_v_h := s_v_h, s_v_t := s_v_t, s_v_d := s_v_d, scale := scale, B := B, H := H, T := T, K := K, V := V, BTL := BTL, BTS := BTS, BK := BK, BV := BV]

end
end VeriTile.Bench.Repairs.ParallelAttentionKernel

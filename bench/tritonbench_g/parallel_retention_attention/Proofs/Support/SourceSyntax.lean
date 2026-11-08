import bench.tritonbench_g.parallel_retention_attention.Kernels

namespace VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages
open VeriTile.Triton FP.Structural
noncomputable section

def forwardSource
    (q k v o : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  forwardPastedKernel q k v o s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d B H T scale BTL BTS BK BV K V

def backwardSource
    (q k v do_ dk dv : RegionName)
    (i_bh i_c i_k i_v i_h : Nat)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d : Nat) (scale : ℝ)
    (B H T K V BTL BTS BK BV : Nat) : ComputeKernel :=
  instantiate_triton_procedure keyValueProcedureSource procedure_args [
    q := q,k := k,v := v,«do» := do_,dk := dk,dv := dv,
    i_bh := i_bh,i_c := i_c,i_k := i_k,i_v := i_v,i_h := i_h,
    s_qk_h := s_qk_h,s_qk_t := s_qk_t,s_qk_d := s_qk_d,
    s_vo_h := s_vo_h,s_vo_t := s_vo_t,s_vo_d := s_vo_d,
    B := B,H := H,T := T,scale := scale,BTL := BTL,BTS := BTS,BK := BK,BV := BV,DK := K,DV := V
  ]
end
end VeriTile.Bench.Repairs.ParallelRetentionAttention.Kernel.FPStages

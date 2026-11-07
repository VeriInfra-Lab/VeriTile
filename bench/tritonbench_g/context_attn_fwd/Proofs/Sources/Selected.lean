import bench.tritonbench_g.context_attn_fwd.Kernels

/-! Invocation adapters to the single pasted repair and retained scale-order candidate. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource
open VeriTile.Triton
noncomputable section
def original (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  originalKernel .fp32 Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N

def scaleFirst (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kb stride_kh stride_ks stride_kd
      stride_vb stride_vh stride_vs stride_vd
      stride_obs stride_oh stride_od
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  scaleFirstKernel .fp32 Q K V sm_scale Out B_Start_Loc B_Seqlen b_prompt_cache_len stride_qbs stride_qh stride_qd stride_kb stride_kh stride_ks stride_kd stride_vb stride_vh stride_vs stride_vd stride_obs stride_oh stride_od kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N

end
end VeriTile.Bench.Repairs.ContextAttnFwd.SelectedSource

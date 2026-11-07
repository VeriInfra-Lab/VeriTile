import bench.tritonbench_g.context_attn_llama.Kernels

/-! Typed entry adapters to the single copied repaired source and existing scale-order candidate. -/
namespace VeriTile.Bench.Repairs.ContextAttnLlama.SelectedSource
open VeriTile.Triton
noncomputable section
def original (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  originalKernel .fp32 Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N

def scaleFirst (Q K V : RegionName) (sm_scale : ℝ) (Out : RegionName)
    (B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N : Nat) : ComputeKernel :=
  scaleFirstKernel .fp32 Q K V sm_scale Out B_Start_Loc B_Seqlen Req_to_tokens B_req_idx b_prompt_cache_len stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num H BLOCK_DMODEL BLOCK_M BLOCK_N

end
end VeriTile.Bench.Repairs.ContextAttnLlama.SelectedSource

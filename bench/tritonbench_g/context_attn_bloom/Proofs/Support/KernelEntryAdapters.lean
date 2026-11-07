import bench.tritonbench_g.context_attn_bloom.Kernels
import VeriTile.Triton.Source
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ContextAttnBloomSource
open VeriTile.Triton
noncomputable section
def kernel :=
  fun
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  => originalKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num b_prompt_cache_len head_dim BLOCK_M BLOCK_DMODEL BLOCK_N

def indexShared :=
  fun
    (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  => indexSharedKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num b_prompt_cache_len head_dim BLOCK_M BLOCK_DMODEL BLOCK_N

def storageProfile (dtype : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Storage) (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  : ComputeKernel := originalKernel dtype Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num b_prompt_cache_len head_dim BLOCK_M BLOCK_DMODEL BLOCK_N

def storageScaleFirst (dtype : VeriTile.Bench.TritonBenchG.ContextAttnBloom.Storage) (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  : ComputeKernel := scaleFirstKernel dtype Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num b_prompt_cache_len head_dim BLOCK_M BLOCK_DMODEL BLOCK_N

def scaleFirst32 (Q K V : RegionName) (sm_scale : ℝ)
    (B_Start_Loc B_Seqlen : RegionName) (Out : RegionName)
    (Req_to_tokens B_req_idx b_prompt_cache_len : RegionName)
    (stride_qbs stride_qh stride_qd
      stride_kbs stride_kh stride_kd
      stride_vbs stride_vh stride_vd
      stride_obs stride_oh stride_od
      stride_req_to_tokens_b stride_req_to_tokens_s
      kv_group_num head_dim BLOCK_M BLOCK_DMODEL BLOCK_N : Nat)  : ComputeKernel := scaleFirstKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out Req_to_tokens B_req_idx stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od stride_req_to_tokens_b stride_req_to_tokens_s kv_group_num b_prompt_cache_len head_dim BLOCK_M BLOCK_DMODEL BLOCK_N

end
end VeriTile.Bench.Repairs.ContextAttnBloomSource

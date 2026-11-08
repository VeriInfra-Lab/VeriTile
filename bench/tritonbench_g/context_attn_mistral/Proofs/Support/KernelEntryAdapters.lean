import bench.tritonbench_g.context_attn_mistral.Kernels
import VeriTile.Triton.Source

/-! Thin invocation adapters to the single repaired source and its two existing
candidates. The finite compatibility storage cases share the pasted bodies. -/
namespace VeriTile.Bench.Repairs.ContextAttnMistralKernel
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.ContextAttnMistral
noncomputable section
def kernel (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num : Nat) (sliding_window : Int)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  originalKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N

def blockStartReused (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num : Nat) (sliding_window : Int)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  blockStartReusedKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N

def storageProfile (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num : Nat) (sliding_window : Int)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  originalKernel dtype Q K V sm_scale B_Start_Loc B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N

def storageScaleFirst (dtype : Storage) (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num : Nat) (sliding_window : Int)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  scaleFirstKernel dtype Q K V sm_scale B_Start_Loc B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N

def scaleFirst32 (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od
        kv_group_num : Nat) (sliding_window : Int)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  scaleFirstKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_M BLOCK_DMODEL BLOCK_N

end
end VeriTile.Bench.Repairs.ContextAttnMistralKernel

import bench.tritonbench_g.context_attn_nopad.Kernels

namespace VeriTile.Bench.Repairs.ContextAttnNopad.SelectedSource
open VeriTile.Triton
noncomputable section
def original (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od : Nat)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  originalKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL BLOCK_N

def scaleFirst (Q K V : RegionName) (sm_scale : ℝ)
      (B_Start_Loc B_Seqlen Out : RegionName)
      (stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd
        stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od : Nat)
      (BLOCK_M BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  scaleFirstKernel .fp32 Q K V sm_scale B_Start_Loc B_Seqlen Out stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od BLOCK_M BLOCK_DMODEL BLOCK_N

end
end VeriTile.Bench.Repairs.ContextAttnNopad.SelectedSource

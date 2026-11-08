import bench.tritonbench_g.attention_fwd_triton2.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.TypedSource
open VeriTile.Triton
noncomputable section

def repaired
    (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N _STAGE : Nat) :
    ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionFwdTriton2.originalKernel q k v qScale kScale out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N _STAGE

end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Whole001.TypedSource

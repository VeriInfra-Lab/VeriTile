import bench.tritonbench_g.attention_forward_triton.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.AttentionForwardTriton.TypedSource
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
  VeriTile.Bench.Repairs.AttentionForwardTriton.originalKernel q k v qScale kScale out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N _STAGE

def hoistedHeadMask
    (q k v qScale kScale out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N _STAGE : Nat) :
    ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionForwardTriton.candidateKernel q k v qScale kScale out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on _Z H N_CTX HEAD_DIM BLOCK_M BLOCK_N _STAGE

end
end VeriTile.Bench.Repairs.AttentionForwardTriton.TypedSource

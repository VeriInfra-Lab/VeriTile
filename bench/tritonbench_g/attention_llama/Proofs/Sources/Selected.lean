import bench.tritonbench_g.attention_llama.Kernels
namespace VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TypedSource
open VeriTile.Triton
noncomputable section

def half
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  halfKernel Q K V Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL IS_CAUSAL

def carryFirstHalfKernel
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.carryFirstHalfKernel Q K V Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL IS_CAUSAL

def bytes
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  bytesKernel Q K V Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL IS_CAUSAL

def carryFirstBytesKernel
    (Q K V Out : RegionName)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
  VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.carryFirstBytesKernel Q K V Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL IS_CAUSAL
end
end VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TypedSource

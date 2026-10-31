import bench.tritonbench_g.attention_llama.Sources.Selected
import VeriTile.Triton.DSL.ScalarView
/-! Proof-only literal-scale views of the actual runtime-scalar repair.
No second complete implementation is maintained here. The actual scalar ABI
and execution proofs remain the public repair boundary. -/
namespace VeriTile.Bench.Repairs.AttentionLlama.TypedSource
open VeriTile.Triton
def repaired
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
 source_scalar_view (VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TypedSource.half Q K V Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL IS_CAUSAL) at "sm_scale" := sm_scale
def repairedFP8
    (Q K V Out : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      N_HEAD H N_CTX start_position
      BLOCK_M BLOCK_N BLOCK_DMODEL : Nat) (IS_CAUSAL : Bool) :
    ComputeKernel :=
 source_scalar_view (VeriTile.Bench.Repairs.AttentionLlamaScalarABIChecked.TypedSource.bytes Q K V Out stride_qz stride_qh stride_qm stride_qk stride_kz stride_kh stride_kn stride_kk stride_vz stride_vh stride_vk stride_vn stride_oz stride_oh stride_om stride_on N_HEAD H N_CTX start_position BLOCK_M BLOCK_N BLOCK_DMODEL IS_CAUSAL) at "sm_scale" := sm_scale
end VeriTile.Bench.Repairs.AttentionLlama.TypedSource

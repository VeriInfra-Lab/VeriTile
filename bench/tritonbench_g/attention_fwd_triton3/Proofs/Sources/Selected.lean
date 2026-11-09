import bench.tritonbench_g.attention_fwd_triton3.Kernels

/-! The repaired Python function bodies live in Body. This compatibility
entry only binds the established Nat flag API to the Python Boolean flags.
All execution, Real and FP clients share these same source definitions. -/
namespace VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Quoted
open VeriTile.Triton
noncomputable section

def kernel
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int) :
    ComputeKernel :=
  sourceKernel Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk
    stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn
    stride_oz stride_oh stride_om stride_on
    _Z H H_KV N_CTX ROUND_CTX NKV_CTX
    sliding_window_offset sliding_window_size BLOCK_M BLOCK_DMODEL BLOCK_N
    (IS_EVEN_M != 0) (IS_EVEN_N != 0) (END != 0) (INIT != 0)
    (SLIDING_WINDOW != 0) (COMPLEMENT_SLIDING_WINDOW != 0)

def keyMaskCSE
    (Q K V M Out L : RegionName) (sm_scale : ℝ)
    (stride_qz stride_qh stride_qm stride_qk
      stride_kz stride_kh stride_kn stride_kk
      stride_vz stride_vh stride_vk stride_vn
      stride_oz stride_oh stride_om stride_on
      _Z H H_KV N_CTX ROUND_CTX NKV_CTX
      IS_EVEN_M IS_EVEN_N BLOCK_M BLOCK_DMODEL BLOCK_N END INIT
      SLIDING_WINDOW COMPLEMENT_SLIDING_WINDOW : Nat)
    (sliding_window_offset sliding_window_size : Int) :
    ComputeKernel :=
  kernelCSEKernel Q K V M Out L sm_scale
    stride_qz stride_qh stride_qm stride_qk
    stride_kz stride_kh stride_kn stride_kk
    stride_vz stride_vh stride_vk stride_vn
    stride_oz stride_oh stride_om stride_on
    _Z H H_KV N_CTX ROUND_CTX NKV_CTX
    sliding_window_offset sliding_window_size BLOCK_M BLOCK_DMODEL BLOCK_N
    (IS_EVEN_M != 0) (IS_EVEN_N != 0) (END != 0) (INIT != 0)
    (SLIDING_WINDOW != 0) (COMPLEMENT_SLIDING_WINDOW != 0)

end
end VeriTile.Bench.Repairs.AttentionFwdTriton3ScalarCast.Quoted

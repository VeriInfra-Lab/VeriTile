import bench.tritonbench_g.matmul_dequantize.Kernels
import bench.tritonbench_g.matmul_dequantize.Proofs.Sources.Mathematical_MatmulDequantize
import bench.tritonbench_g.matmul_dequantize.Proofs.Sources.Mathematical_FPDequant
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 24000
/-! All three original accepted stages. Typed source alone is not a numerical
completion claim; independent loop/value contracts are supplied separately. -/

def matmul4Typed (dtype : FloatDType)
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := Body.original_matmul4_kernel dtype a_ptr b_ptr c_ptr scales_ptr zeros_ptr M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n groupsize BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NO_GROUPS

end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped

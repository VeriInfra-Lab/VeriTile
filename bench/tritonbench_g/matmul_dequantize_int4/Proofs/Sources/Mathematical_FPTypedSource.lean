import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Sources.NativeLowering
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.Sources.Mathematical_MatmulDequantizeInt4
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Sources.Mathematical_FPContract
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-! Original accepted loop extent, source statements and branch coverage.
Packed inputs retain int32 width; implicit integer-to-fp16 and output storage
conversions are materialized. A dtype/source declaration alone carries no
execution, numerical correctness, or native dot evidence. -/

def typed (dtype : FloatDType)
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := Source.originalKernel dtype a_ptr b_ptr c_ptr scales_ptr zeros_ptr M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n groupsize BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NO_GROUPS

end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped

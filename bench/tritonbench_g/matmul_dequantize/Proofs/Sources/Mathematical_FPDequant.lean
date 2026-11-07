import bench.tritonbench_g.matmul_dequantize.Kernels
import bench.tritonbench_g.matmul_dequantize.Proofs.Sources.Mathematical_MatmulDequantize
import bench.tritonbench_g.matmul_dequant_int4.Proofs.Sources.Mathematical_FPCorrect
namespace VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 12000
def dequantTyped (dtype : FloatDType)
    (b_ptr b_scale_ptr b_zp_ptr : RegionName)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel := MatmulDequantInt4.Body.dequantizeKernel dtype b_ptr b_scale_ptr b_zp_ptr fpb_ptr K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn stride_fpbk stride_fpbn BLOCK_SIZE_K BLOCK_SIZE_N

namespace Standalone
abbrev Parameters := MatmulDequantInt4.FPTyped.Parameters
abbrev Half := MatmulDequantInt4.Binary16.Value

end Standalone
end
end VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped

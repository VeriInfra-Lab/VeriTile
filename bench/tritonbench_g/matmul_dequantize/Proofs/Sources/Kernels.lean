import bench.tritonbench_g.matmul_dequantize.Kernels
import bench.tritonbench_g.matmul_dequantize.Proofs.Sources.Mathematical_FPTypedSource
import bench.tritonbench_g.matmul_dequantize.Proofs.Sources.Mathematical_FPDequant
namespace VeriTile.Bench.Repairs.MatmulDequantizeWhole001
open VeriTile.Triton
noncomputable section
namespace FPTyped

abbrev dequantTyped := VeriTile.Bench.TritonBenchG.MatmulDequantize.FPTyped.dequantTyped
def matmulTyped (dtype : FloatDType)
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn
      group_size : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := VeriTile.Bench.TritonBenchG.MatmulDequantize.Body.original_matmul_kernel dtype a_ptr b_ptr c_ptr bs_ptr bzp_ptr M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_bsk stride_bsn stride_bzpk stride_bzpn group_size BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M 1
end FPTyped
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001

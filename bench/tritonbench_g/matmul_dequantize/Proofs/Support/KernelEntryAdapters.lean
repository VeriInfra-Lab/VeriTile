import bench.tritonbench_g.matmul_dequantize.Proofs.Sources.Kernels

/-! The original readable quotations are shared with all reused proofs:
`Whole001/Kernels.lean` supplies the repaired s2 body and reexports the original
GPTQ/dequantization quotations. Complete concrete candidates below preserve
those selected source precisions and connect to callable Python candidates. -/
namespace VeriTile.Bench.Repairs.MatmulDequantize.KernelSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
noncomputable section

/-- Complete concrete candidates; precomputed scalar origins reduce runtime
integer address arithmetic without changing floating computation. -/
def groupedGPTQ (dtype : FloatDType)
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := VeriTile.Bench.TritonBenchG.MatmulDequantize.Body.candidate_matmul4_kernel dtype a_ptr b_ptr c_ptr scales_ptr zeros_ptr M K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n groupsize N BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M NO_GROUPS

def groupedS2 (dtype : FloatDType)
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn
      group_size : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := VeriTile.Bench.TritonBenchG.MatmulDequantize.Body.candidate_matmul_kernel dtype a_ptr b_ptr c_ptr bs_ptr bzp_ptr M K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn stride_bsk stride_bsn stride_bzpk stride_bzpn group_size N BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M 1

def affineDequant (dtype : FloatDType)
    (b_ptr b_scale_ptr b_zp_ptr : RegionName)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel := VeriTile.Bench.TritonBenchG.MatmulDequantize.Body.candidate_dequantize_kernel dtype b_ptr b_scale_ptr b_zp_ptr fpb_ptr K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn stride_fpbk stride_fpbn BLOCK_SIZE_K BLOCK_SIZE_N

end
end VeriTile.Bench.Repairs.MatmulDequantize.KernelSource

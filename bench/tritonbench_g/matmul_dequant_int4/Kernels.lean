import VeriTile.Triton.DSL.PackedIntegerPromotion
import bench.tritonbench_g.matmul_dequantize_int4.Kernels
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4.Body
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 5000000
/-! Complete standalone repaired source and the existing contiguous-output
candidate. The unlaunched fused JIT is identical to the shared source at
MatmulDequantizeInt4.Source.originalBody, registered directly in the body manifest. -/
def dequantizeSignature (dtype : FloatDType) : CallSignature where
  pointers := [("b_ptr",.i32),("b_zp_ptr",.i32)]
  floatingPointers := [("b_scale_ptr",dtype),("fpb_ptr",dtype)]
  scalars := [("K",.integer), ("N",.integer), ("group_size",.integer), ("stride_bk",.integer), ("stride_bn",.integer), ("stride_bsk",.integer), ("stride_bsn",.integer), ("stride_bzpk",.integer), ("stride_bzpn",.integer), ("stride_fpbk",.integer), ("stride_fpbn",.integer)]
  constexpr := ["BLOCK_SIZE_K","BLOCK_SIZE_N"]
def dequantizeKernel (dtype : FloatDType) (b_ptr b_scale_ptr b_zp_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn stride_fpbk stride_fpbn BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel :=
  with_triton_call (dequantizeSignature dtype) call_args [K := K, N := N, group_size := group_size, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsk := stride_bsk, stride_bsn := stride_bsn, stride_bzpk := stride_bzpk, stride_bzpn := stride_bzpn, stride_fpbk := stride_fpbk, stride_fpbn := stride_fpbn] in triton {
      """Dequantize tile [BLOCK_SIZE_K, BLOCK_SIZE_N] in full precision.
      We should assert BLOCK_SIZE_N % 8 == 0.
      weight[K // 8, N], scale[K // group_size, N], zp[K // group_size, N // group_size]
      """
      k_block_idx = tl.program_id(axis=0)
      n_block_idx = tl.program_id(axis=1)
      offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
      offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      fpb_offs = offs_k[:, None] * stride_fpbk + offs_n[None, :] * stride_fpbn
      b_offs = (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
      bzp_offs = (offs_k[:, None] // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
      bs_offs = (offs_k[:, None] // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
      n_mask = offs_n[None, :] < N
      k_mask = offs_k[:, None] < K
      mask = n_mask & k_mask
      int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
      zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
      scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
      b_shift = (offs_k[:, None] % 8) * 4
      bzp_shift = (offs_n[None, :] % 8) * 4
      fp_weight = (((int32_b >> b_shift) & 0xF) - ((zp_b >> bzp_shift) & 0xF)) * scale_b
      tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
  }

def unitDequantizeKernel (dtype : FloatDType) (b_ptr b_scale_ptr b_zp_ptr fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn stride_bzpk stride_bzpn stride_fpbk stride_fpbn BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel :=
  with_triton_call (dequantizeSignature dtype) call_args [K := K, N := N, group_size := group_size, stride_bk := stride_bk, stride_bn := stride_bn, stride_bsk := stride_bsk, stride_bsn := stride_bsn, stride_bzpk := stride_bzpk, stride_bzpn := stride_bzpn, stride_fpbk := stride_fpbk, stride_fpbn := stride_fpbn] in triton {
      """Dequantize tile [BLOCK_SIZE_K, BLOCK_SIZE_N] in full precision.
      We should assert BLOCK_SIZE_N % 8 == 0.
      weight[K // 8, N], scale[K // group_size, N], zp[K // group_size, N // group_size]
      """
      k_block_idx = tl.program_id(axis=0)
      n_block_idx = tl.program_id(axis=1)
      offs_k = k_block_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
      offs_n = n_block_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
      fpb_offs = offs_k[:, None] * stride_fpbk + offs_n[None, :]
      b_offs = (offs_k[:, None] // 8) * stride_bk + offs_n[None, :] * stride_bn
      bzp_offs = (offs_k[:, None] // group_size) * stride_bzpk + (offs_n[None, :] // 8) * stride_bzpn
      bs_offs = (offs_k[:, None] // group_size) * stride_bsk + offs_n[None, :] * stride_bsn
      n_mask = offs_n[None, :] < N
      k_mask = offs_k[:, None] < K
      mask = n_mask & k_mask
      int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
      zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
      scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
      b_shift = (offs_k[:, None] % 8) * 4
      bzp_shift = (offs_n[None, :] % 8) * 4
      fp_weight = (((int32_b >> b_shift) & 0xF) - ((zp_b >> bzp_shift) & 0xF)) * scale_b
      tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
  }

end
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4.Body

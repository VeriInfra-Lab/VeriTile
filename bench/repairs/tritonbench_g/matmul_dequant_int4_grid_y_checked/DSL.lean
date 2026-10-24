import bench.repairs.tritonbench_g.matmul_dequant_int4_grid_y_checked.Proofs.Source
import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPReference

namespace VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.DSL
open VeriTile.Triton
noncomputable section

/-! The two complete source quotations in one readable module. Their syntax
is unchanged; the binding equalities below are definitional, not algorithmic
transformations or numerical equivalences. -/

namespace Standalone
def typed (dtype : FloatDType)
    (b_ptr b_scale_ptr b_zp_ptr : RegionName)
    (fpb_ptr : RegionName)
    (K N group_size stride_bk stride_bn stride_bsk stride_bsn
      stride_bzpk stride_bzpn stride_fpbk stride_fpbn : Nat)
    (BLOCK_SIZE_K BLOCK_SIZE_N : Nat) : ComputeKernel := triton {
  tl.region b_ptr = tl.int32, b_zp_ptr = tl.int32, b_scale_ptr = DTYPE, fpb_ptr = DTYPE
  k_block_idx = tl.program_id(axis=0)
  n_block_idx = tl.program_id(axis=1)
  offs_k = k_block_idx * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  offs_n = n_block_idx * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  fpb_offs = offs_k[:, None] * $(stride_fpbk) + offs_n[None, :] * $(stride_fpbn)
  b_offs = (offs_k[:, None] // $(8)) * $(stride_bk) + offs_n[None, :] * $(stride_bn)
  bzp_offs = (offs_k[:, None] // $(group_size)) * $(stride_bzpk) + (offs_n[None, :] // $(8)) * $(stride_bzpn)
  bs_offs = (offs_k[:, None] // $(group_size)) * $(stride_bsk) + offs_n[None, :] * $(stride_bsn)
  n_mask = offs_n[None, :] < $(N)
  k_mask = offs_k[:, None] < $(K)
  mask = n_mask & k_mask
  int32_b = tl.load(b_ptr + b_offs, mask=mask, other=0.0)
  zp_b = tl.load(b_zp_ptr + bzp_offs, mask=mask, other=0.0)
  scale_b = tl.load(b_scale_ptr + bs_offs, mask=mask, other=0.0)
  b_shift = (offs_k[:, None] % $(8)) * $(4)
  bzp_shift = (offs_n[None, :] % $(8)) * $(4)
  fp_weight = tl.toReal(((int32_b >> b_shift) & 15) - ((zp_b >> bzp_shift) & 15)).to(DTYPE) * scale_b
  tl.store(fpb_ptr + fpb_offs, fp_weight, mask=mask)
}

end Standalone
namespace Fused
def typed (dtype : FloatDType)
    (a_ptr c_ptr scales_ptr : RegionName) (b_ptr zeros_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_scales_g stride_scales_n stride_zeros_g stride_zeros_n
      groupsize : Nat) (NO_GROUPS : Bool)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M : Nat) :
    ComputeKernel := triton {
  tl.region a_ptr = DTYPE, c_ptr = DTYPE, scales_ptr = DTYPE, b_ptr = tl.int32, zeros_ptr = tl.int32

  bits = $(4)
  infearure_per_bits = $(8)
  pid = tl.program_id(axis=0)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_bn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  offs_k = tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  a_mask = (offs_am[:, None] < $(M))
  b_ptrs = b_ptr + ((offs_k[:, None] // infearure_per_bits) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  scales_ptrs = scales_ptr + offs_bn * $(stride_scales_n)
  zeros_ptrs = zeros_ptr + ((offs_bn // infearure_per_bits) * $(stride_zeros_n))
  shifter = (offs_k % infearure_per_bits) * bits
  zeros_shifter = (offs_bn % infearure_per_bits) * bits
  if NO_GROUPS {
    scales = tl.load(scales_ptrs)
    zeros = tl.load(zeros_ptrs)
    zeros = (zeros >> zeros_shifter) & 15
    zeros = tl.toReal(zeros).to(DTYPE) * scales
  }
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), num_pid_k, $(1)) {
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs)
    if not NO_GROUPS {
      g_id = k // ($(groupsize) // $(BLOCK_SIZE_K))
      ptr = scales_ptrs + g_id * $(stride_scales_g)
      scales = tl.load(ptr)
      ptr = zeros_ptrs + g_id * $(stride_zeros_g)
      zeros = tl.load(ptr)
      zeros = (zeros >> zeros_shifter) & 15
      zeros = tl.toReal(zeros).to(DTYPE) * scales
    }
    b = (b >> shifter[:, None]) & 15
    b = tl.toReal(b).to(DTYPE) * scales[None, :] - zeros[None, :]
    accumulator += tl.dot(a, (b).to(a.dtype))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(c_ptr.dtype.element_ty)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, accumulator.to(DTYPE), mask=c_mask)
}

end Fused

theorem standalone_binding (f : Source.Profile) (p : Source.Parameters) :
    Source.source f p=Standalone.typed f.dtype p.weight p.scale p.zero p.output
      p.rows p.columns p.groupSize p.weightRowStride p.weightColumnStride
      p.scaleRowStride p.scaleColumnStride p.zeroRowStride p.zeroColumnStride
      p.outputRowStride p.outputColumnStride p.blockRows p.blockColumns := rfl

theorem fused_binding (p : VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped.Parameters) :
    VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPTyped.source p=
      Fused.typed .fp16 p.input p.output p.scale p.weight p.zero
        p.rows p.columns p.inner p.inputRowStride p.inputInnerStride p.weightInnerStride p.weightColumnStride
        p.outputRowStride p.outputColumnStride p.scaleGroupStride p.scaleColumnStride
        p.zeroGroupStride p.zeroColumnStride p.groupSize p.noGroups p.blockRows p.blockColumns p.blockInner p.groupRows := rfl

#axiomsClean standalone_binding
#axiomsClean fused_binding
end
end VeriTile.Bench.Repairs.MatmulDequantInt4GridYChecked.DSL

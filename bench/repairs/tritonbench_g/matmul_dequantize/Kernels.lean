import bench.repairs.tritonbench_g.matmul_dequantize.Whole001.Kernels

/-! The original readable quotations are shared with all reused proofs:
`Whole001/Kernels.lean` supplies the repaired s2 body and reexports the original
GPTQ/dequantization quotations. Complete concrete candidates below preserve
those selected source precisions and connect to callable Python candidates. -/
namespace VeriTile.Bench.Repairs.MatmulDequantize.KernelSource
open VeriTile.Triton
open VeriTile.Bench.Repairs.MatmulDequantizeWhole001
noncomputable section

/-- Both checked GPTQ configurations use FP16 operands/storage and FP32 dot
accumulation. The quotation makes the implicit FP16 store conversion explicit. -/
abbrev gptq := FPTyped.matmul4Typed .fp16

/-- Canonical repaired SPLIT_K=1 body: FP32 A/scales, all four K-tail masks,
FP16 dot operands, FP32 accumulation and FP16 output. -/
abbrev s2 := FPTyped.matmulTyped .fp32

/-- The same full source body is selected at its requested FP16/FP32 precision.
The FP16 route is also the Triton part of s1; torch.mm is an external consumer. -/
abbrev dequant16 := FPTyped.dequantTyped .fp16
abbrev dequant32 := FPTyped.dequantTyped .fp32
/-- Complete concrete candidates; precomputed scalar origins reduce runtime
integer address arithmetic without changing floating computation. -/
def groupedGPTQ (dtype : FloatDType)
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
  num_pid_in_group = $(GROUP_SIZE_M * ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N))
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
    accumulator += tl.dot(a, b)
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) // infearure_per_bits) * $(stride_bk)
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, accumulator.to(DTYPE), mask=c_mask)
}


def groupedS2 (dtype : FloatDType)
    (a_ptr c_ptr bs_ptr : RegionName) (b_ptr bzp_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn
      stride_bsk stride_bsn stride_bzpk stride_bzpn
      group_size : Nat)
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.region a_ptr = DTYPE, c_ptr = tl.float16, bs_ptr = DTYPE, b_ptr = tl.int32, bzp_ptr = tl.int32

  pid = tl.program_id(axis=0)
  pid_sp_k = tl.program_id(axis=1)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M * ((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N))
  group_id = pid // num_pid_in_group
  first_pid_m = group_id * $(GROUP_SIZE_M)
  group_size_m = min(num_pid_m - first_pid_m, $(GROUP_SIZE_M))
  pid_m = first_pid_m + (pid % group_size_m)
  pid_n = (pid % num_pid_in_group) // group_size_m
  offs_am = (pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))) % $(M)
  offs_bn = (pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))) % $(N)
  offs_k = pid_sp_k * $(BLOCK_SIZE_K) + tl.arange(0, $(BLOCK_SIZE_K))
  a_ptrs = a_ptr + (offs_am[:, None] * $(stride_am) + offs_k[None, :] * $(stride_ak))
  b_ptrs = b_ptr + ((offs_k[:, None] // $(8)) * $(stride_bk) + offs_bn[None, :] * $(stride_bn))
  accumulator = tl.zeros([$(BLOCK_SIZE_M), $(BLOCK_SIZE_N)], dtype=tl.float32)
  for k in range($(0), $(numKBlocks), $(1)) {
    bs_ptrs = bs_ptr + (((offs_k[:, None] + k * $(BLOCK_SIZE_K)) // $(group_size)) * $(stride_bsk) + offs_bn[None, :] * $(stride_bsn))
    bzp_ptrs = bzp_ptr + (((offs_k[:, None] + k * $(BLOCK_SIZE_K)) // $(group_size)) * $(stride_bzpk) + (offs_bn[None, :] // $(8)) * $(stride_bzpn))
    b_shift_bits = (offs_k[:, None] % $(8)) * $(4)
    bzp_shift_bits = (offs_bn[None, :] % $(8)) * $(4)
    a = tl.load(a_ptrs, mask=(offs_k[None, :] + k * $(BLOCK_SIZE_K) < $(K)), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * $(BLOCK_SIZE_K) < $(K)), other=0.0)
    bs = tl.load(bs_ptrs, mask=(offs_k[:, None] + k * $(BLOCK_SIZE_K) < $(K)), other=0.0)
    bzp = tl.load(bzp_ptrs, mask=(offs_k[:, None] + k * $(BLOCK_SIZE_K) < $(K)), other=0.0)
    int_b = (b >> b_shift_bits) & 15
    int_bzp = (bzp >> bzp_shift_bits) & 15
    b = (tl.toReal(int_b - int_bzp).to(DTYPE) * bs).to(tl.float16)
    accumulator += tl.dot((a).to(tl.float16), (b).to(tl.float16))
    a_ptrs += $(BLOCK_SIZE_K) * $(stride_ak)
    b_ptrs += ($(BLOCK_SIZE_K) * $(stride_bk) // $(8))
  }
  c = (accumulator).to(tl.float16)
  offs_cm = pid_m * $(BLOCK_SIZE_M) + tl.arange(0, $(BLOCK_SIZE_M))
  offs_cn = pid_n * $(BLOCK_SIZE_N) + tl.arange(0, $(BLOCK_SIZE_N))
  c_ptrs = c_ptr + $(stride_cm) * offs_cm[:, None] + $(stride_cn) * offs_cn[None, :]
  c_mask = (offs_cm[:, None] < $(M)) & (offs_cn[None, :] < $(N))
  tl.store(c_ptrs, c, mask=c_mask)
}

def affineDequant (dtype : FloatDType)
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
  fpb_offs = k_block_idx * $(BLOCK_SIZE_K * stride_fpbk) + n_block_idx * $(BLOCK_SIZE_N * stride_fpbn) + tl.arange(0, $(BLOCK_SIZE_K))[:, None] * $(stride_fpbk) + tl.arange(0, $(BLOCK_SIZE_N))[None, :] * $(stride_fpbn)
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

end
end VeriTile.Bench.Repairs.MatmulDequantize.KernelSource

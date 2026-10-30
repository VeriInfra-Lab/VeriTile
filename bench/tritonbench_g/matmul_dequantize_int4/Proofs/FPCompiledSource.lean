import bench.tritonbench_g.matmul_dequantize_int4.Proofs.FPSoftware
import bench.tritonbench_g.matmul_dequant_int4.Proofs.FPContract
namespace VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000

/-! Explicit compiled live-carry dot graph. This differs from the old source graph
only at the accumulator assignment; no equality between those graphs is assumed.
Original accepted loop extent, source statements and branch coverage.
Packed inputs retain int32 width; implicit integer-to-fp16 and output storage
conversions are materialized. A dtype/source declaration alone carries no
execution, numerical correctness, or native dot evidence. -/

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
    accumulator = tl.dot(a, (b).to(a.dtype), accumulator)
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

end
end VeriTile.Bench.TritonBenchG.MatmulDequantizeInt4.FPCompiled

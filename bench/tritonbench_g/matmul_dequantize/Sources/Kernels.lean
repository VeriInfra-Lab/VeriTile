import bench.tritonbench_g.matmul_dequantize.Sources.Mathematical_FPTypedSource
import bench.tritonbench_g.matmul_dequantize.Sources.Mathematical_FPDequant
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
    (BLOCK_SIZE_M BLOCK_SIZE_N BLOCK_SIZE_K GROUP_SIZE_M numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.region a_ptr = DTYPE, c_ptr = tl.float16, bs_ptr = DTYPE, b_ptr = tl.int32, bzp_ptr = tl.int32

  pid = tl.program_id(axis=0)
  pid_sp_k = tl.program_id(axis=1)
  num_pid_m = tl.cdiv($(M), $(BLOCK_SIZE_M))
  num_pid_n = tl.cdiv($(N), $(BLOCK_SIZE_N))
  num_pid_k = tl.cdiv($(K), $(BLOCK_SIZE_K))
  num_pid_in_group = $(GROUP_SIZE_M) * num_pid_n
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
end FPTyped
end
end VeriTile.Bench.Repairs.MatmulDequantizeWhole001

import bench.tritonbench_g.int8_dequant_matmul.Int8DequantMatmul
import VeriTile.Triton.Float.IntegerDot
import bench.tritonbench_g.int4_matmul.Proofs.FPMixed
namespace VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 10000000
set_option maxRecDepth 32000
/-! Faithful typed transcription of the existing SPLIT_K=1, EVEN_K=true
scope. Both has_bias branches remain. Runtime scalar parameter bits and
integer width boundaries are made explicit in FPSource.lean. -/
def typed
    (a_ptr b_ptr c_ptr bias_ptr state_x_ptr state_w_ptr : RegionName)
    (M N K : Nat) (divfactor : ℝ) (has_bias : Bool)
    (stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M numKBlocks : Nat) :
    ComputeKernel := triton {
  tl.region a_ptr = tl.int8, b_ptr = tl.int8, c_ptr = tl.float16, bias_ptr = tl.float16, state_x_ptr = tl.float32, state_w_ptr = tl.float32
  pid = tl.program_id(0)
  pid_z = tl.program_id(1)
  grid_m = tl.cdiv($(M), $(BLOCK_M))
  grid_n = tl.cdiv($(N), $(BLOCK_N))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = tl.max_contiguous(tl.multiple_of(rm % $(M), $(BLOCK_M)), $(BLOCK_M))
  rbn = tl.max_contiguous(tl.multiple_of(rn % $(N), $(BLOCK_N)), $(BLOCK_N))
  rk = pid_z * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  A = a_ptr + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = b_ptr + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  w_factor = tl.load(state_w_ptr + rbn)[None, :]
  x_factor = tl.load(state_x_ptr + ram)[:, None]
  acc_int = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range($(0), $(numKBlocks), $(1)) {
    a = tl.load(A)
    b = tl.load(B)
    acc_int += tl.dot(a, b)
    A += $(BLOCK_K) * $(stride_ak)
    B += $(BLOCK_K) * $(stride_bk)
  }
  scaled = w_factor * (x_factor * (acc_int * $((divfactor : ℝ))))
  out = (scaled).to(tl.float16)
  if has_bias {
    bias = tl.load(bias_ptr + rn).to(tl.float16)
    out = out + bias[None, :]
  }
  C = c_ptr + (rm[:, None] * $(stride_cm) + rn[None, :] * $(stride_cn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  tl.store(C, out, mask=mask)
}

end
end VeriTile.Bench.TritonBenchG.Int8DequantMatmul.FPTyped

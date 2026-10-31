import VeriTile.Triton.Source
import VeriTile.Triton.DSL

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.TypedSource
open Lean VeriTile.Triton

/-- Complete scalar-cast candidate JIT: all SPLIT_K, EVEN_K and bias branches retained.
Index expressions use the exact natural descriptor specialization after the
explicit casts; Machine.RuntimeIndex.exact and IndexABI.formed justify that
normalization. Signed cdiv is also supported by the shared frontend. The fixed output dtype is the original host's FP16; public scale/bias inputs
use FP32. Raw runtime scalar representation is retained by the host ABI. -/
def kernel (a_ptr b_ptr c_ptr bias_ptr state_x_ptr state_w_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K steps : Nat)
    (EVEN_K has_bias : Bool) : ComputeKernel :=
  triton {
  tl.ports inputs=[a_ptr,b_ptr,bias_ptr,state_x_ptr,state_w_ptr], outputs=[c_ptr]
  tl.region a_ptr = tl.int8, b_ptr = tl.int8, c_ptr = tl.float16, bias_ptr = tl.float32, state_x_ptr = tl.float32, state_w_ptr = tl.float32
  tl.param divfactor : tl.float32 (carrier=tl.float32)
  pid = tl.program_id(0).to(tl.int64)
  pid_z = tl.program_id(1).to(tl.int64)
  M64 = tl.cast($(M), tl.int64)
  N64 = tl.cast($(N), tl.int64)
  K64 = tl.cast($(K), tl.int64)
  stride_am64 = tl.cast($(stride_am), tl.int64)
  stride_ak64 = tl.cast($(stride_ak), tl.int64)
  stride_bk64 = tl.cast($(stride_bk), tl.int64)
  stride_bn64 = tl.cast($(stride_bn), tl.int64)
  stride_cm64 = tl.cast($(stride_cm), tl.int64)
  stride_cn64 = tl.cast($(stride_cn), tl.int64)
  grid_m = tl.cdiv($(M), $(BLOCK_M))
  grid_n = tl.cdiv($(N), $(BLOCK_N))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
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
  for k in range($(0), $(steps), $(1)) {
    if EVEN_K {
      a = tl.load(A)
      b = tl.load(B)
    } else {
      k_remaining = $(K) - k * $(BLOCK_K * SPLIT_K)
      a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
      b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
    }
    acc_int += tl.dot(a, b)
    A += $(BLOCK_K * SPLIT_K) * $(stride_ak)
    B += $(BLOCK_K * SPLIT_K) * $(stride_bk)
  }
  scaled = w_factor * (x_factor * (acc_int * divfactor))
  out = scaled.to(tl.float16)
  if has_bias {
    if pid_z == $(0) {
      bias_value = tl.load(bias_ptr + rn, mask=rn < $(N), other=0.0).to(tl.float16)
      out = out + bias_value[None, :]
    }
  }
  C = c_ptr + (rm[:, None] * $(stride_cm) + rn[None, :] * $(stride_cn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  if SPLIT_K == 1 {
    tl.store(C, out, mask=mask)
  } else {
    tl.atomic_add(C, out, mask=mask)
  }
}

/-- Candidate reuses the first rm/rn values instead of recomputing the
same index tiles after A/B pointer construction. No numerical operation moves. -/
def reuseIndicesKernel (a_ptr b_ptr c_ptr bias_ptr state_x_ptr state_w_ptr : RegionName)
    (M N K stride_am stride_ak stride_bk stride_bn stride_cm stride_cn : Nat)
    (BLOCK_M BLOCK_N BLOCK_K GROUP_M SPLIT_K steps : Nat)
    (EVEN_K has_bias : Bool) : ComputeKernel :=
  triton {
  tl.ports inputs=[a_ptr,b_ptr,bias_ptr,state_x_ptr,state_w_ptr], outputs=[c_ptr]
  tl.region a_ptr = tl.int8, b_ptr = tl.int8, c_ptr = tl.float16, bias_ptr = tl.float32, state_x_ptr = tl.float32, state_w_ptr = tl.float32
  tl.param divfactor : tl.float32 (carrier=tl.float32)
  pid = tl.program_id(0).to(tl.int64)
  pid_z = tl.program_id(1).to(tl.int64)
  M64 = tl.cast($(M), tl.int64)
  N64 = tl.cast($(N), tl.int64)
  K64 = tl.cast($(K), tl.int64)
  stride_am64 = tl.cast($(stride_am), tl.int64)
  stride_ak64 = tl.cast($(stride_ak), tl.int64)
  stride_bk64 = tl.cast($(stride_bk), tl.int64)
  stride_bn64 = tl.cast($(stride_bn), tl.int64)
  stride_cm64 = tl.cast($(stride_cm), tl.int64)
  stride_cn64 = tl.cast($(stride_cn), tl.int64)
  grid_m = tl.cdiv($(M), $(BLOCK_M))
  grid_n = tl.cdiv($(N), $(BLOCK_N))
  width = $(GROUP_M) * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * $(GROUP_M), $(GROUP_M))
  pid_m = group_id * $(GROUP_M) + (pid % group_size)
  pid_n = (pid % width) // group_size
  rm = pid_m * $(BLOCK_M) + tl.arange(0, $(BLOCK_M))
  rn = pid_n * $(BLOCK_N) + tl.arange(0, $(BLOCK_N))
  ram = tl.max_contiguous(tl.multiple_of(rm % $(M), $(BLOCK_M)), $(BLOCK_M))
  rbn = tl.max_contiguous(tl.multiple_of(rn % $(N), $(BLOCK_N)), $(BLOCK_N))
  rk = pid_z * $(BLOCK_K) + tl.arange(0, $(BLOCK_K))
  A = a_ptr + (ram[:, None] * $(stride_am) + rk[None, :] * $(stride_ak))
  B = b_ptr + (rk[:, None] * $(stride_bk) + rbn[None, :] * $(stride_bn))
  w_factor = tl.load(state_w_ptr + rbn)[None, :]
  x_factor = tl.load(state_x_ptr + ram)[:, None]
  acc_int = tl.zeros([$(BLOCK_M), $(BLOCK_N)], dtype=tl.int32)
  for k in range($(0), $(steps), $(1)) {
    if EVEN_K {
      a = tl.load(A)
      b = tl.load(B)
    } else {
      k_remaining = $(K) - k * $(BLOCK_K * SPLIT_K)
      a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
      b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)
    }
    acc_int += tl.dot(a, b)
    A += $(BLOCK_K * SPLIT_K) * $(stride_ak)
    B += $(BLOCK_K * SPLIT_K) * $(stride_bk)
  }
  scaled = w_factor * (x_factor * (acc_int * divfactor))
  out = scaled.to(tl.float16)
  if has_bias {
    if pid_z == $(0) {
      bias_value = tl.load(bias_ptr + rn, mask=rn < $(N), other=0.0).to(tl.float16)
      out = out + bias_value[None, :]
    }
  }
  C = c_ptr + (rm[:, None] * $(stride_cm) + rn[None, :] * $(stride_cn))
  mask = (rm < $(M))[:, None] & (rn < $(N))[None, :]
  if SPLIT_K == 1 {
    tl.store(C, out, mask=mask)
  } else {
    tl.atomic_add(C, out, mask=mask)
  }
}

/- The unused predecessor skeleton has been retired; both complete
programs above use the repaired runtime scalar interface. -/

end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.TypedSource

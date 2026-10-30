import bench.tritonbench_g.sgmv_expand_slice.Sources.Kernels
namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexReuse
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 40000
def sourceHalfSingle323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int32, seq_lens = tl.int32, lora_indices = tl.int32
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceHalfSingle323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int32, seq_lens = tl.int32, lora_indices = tl.int64
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceHalfSingle326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int32, seq_lens = tl.int64, lora_indices = tl.int32
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceHalfSingle326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int32, seq_lens = tl.int64, lora_indices = tl.int64
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceHalfSingle643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int64, seq_lens = tl.int32, lora_indices = tl.int32
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceHalfSingle643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int64, seq_lens = tl.int32, lora_indices = tl.int64
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceHalfSingle646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int64, seq_lens = tl.int64, lora_indices = tl.int32
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceHalfSingle646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel := triton {
  tl.region input_ptr = DTYPE, lora_ptr = tl.float16, out_ptr = tl.float32
  tl.region b_seq_start_loc = tl.int64, seq_lens = tl.int64, lora_indices = tl.int64
  pid = tl.program_id(axis=0).to(tl.int64)
  cur_batch = tl.cast($(batch), tl.int64)
  N = tl.cast($(N), tl.int64)
  K = tl.cast($(K), tl.int64)
  xm_stride = tl.cast($(xm_stride), tl.int64)
  xk_stride = tl.cast($(xk_stride), tl.int64)
  l0_stride = tl.cast($(l0_stride), tl.int64)
  lora_k_stride = tl.cast($(lora_k_stride), tl.int64)
  lora_n_stride = tl.cast($(lora_n_stride), tl.int64)
  cm_stride = tl.cast($(cm_stride), tl.int64)
  cn_stride = tl.cast($(cn_stride), tl.int64)
  slice_offset = tl.cast($(slice_offset), tl.int64)
  cta_n_num = (N + $((32 : Nat)) - $((1 : Nat))) // $((32 : Nat))
  pid_m = pid // cta_n_num
  pid_n = pid % cta_n_num
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  if M <= $((0 : Nat)) {
    return
  }
  if pid_m * $((32 : Nat)) >= M {
    return
  }
  lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
  if lora_index == $((-1 : Int)) {
    return
  }
  cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
  offset_m = tl.arange(0, 32).to(tl.int64) + pid_m * $((32 : Nat))
  offset_n = tl.arange(0, 32).to(tl.int64) + pid_n * $((32 : Nat))
  offset_k = tl.arange(0, 16).to(tl.int64)
  ram = offset_m % M
  rbn = offset_n % N
  accumulator = tl.zeros([32, 32], dtype=tl.float32)
  for k in range((K + $((16 : Nat)) - $((1 : Nat))) // $((16 : Nat))) {
    global_k = k * $((16 : Nat)) + offset_k
    safe_k = tl.where(global_k < K, global_k, $((0 : Nat)))
    a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride + safe_k[None, :] * xk_stride)
    b_ptr = (lora_ptr + l0_stride * lora_index + safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
    if $((evenK : Bool)) {
      tiled_a = tl.load(a_ptr)
      tiled_b = tl.load(b_ptr)
    } else {
      tiled_a = tl.load(a_ptr, mask=global_k[None, :] < K, other=0)
      tiled_b = tl.load(b_ptr, mask=global_k[:, None] < K, other=0)
    }
    if $((castType : Bool)) {
      tiled_a = tiled_a.to(tl.float16)
    }
    accumulator = tl.dot(tiled_a, tiled_b, accumulator)
  }
  tiled_c = accumulator.to(tl.float16)
  offset_cm = cur_seq_start + offset_m
  offset_cn = offset_n + slice_offset
  safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, $((0 : Nat)))
  safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
  c_ptr = (out_ptr + safe_cm[:, None] * cm_stride + safe_cn[None, :] * cn_stride)
  M = tl.load(seq_lens + cur_batch).to(tl.int64)
  c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
  if $((addInputs : Bool)) {
    tiled_out = tl.load(c_ptr, mask=c_mask)
    tiled_c = (tiled_c + tiled_out)
    tl.store(c_ptr, tiled_c, mask=c_mask)
  } else {
    tl.store(c_ptr, tiled_c.to(tl.float32), mask=c_mask)
  }
}

def sourceFace (starts lengths adapters : MetadataWidth) :=
  match starts,lengths,adapters with
  | .w32, .w32, .w32 => sourceHalfSingle323232
  | .w32, .w32, .w64 => sourceHalfSingle323264
  | .w32, .w64, .w32 => sourceHalfSingle326432
  | .w32, .w64, .w64 => sourceHalfSingle326464
  | .w64, .w32, .w32 => sourceHalfSingle643232
  | .w64, .w32, .w64 => sourceHalfSingle643264
  | .w64, .w64, .w32 => sourceHalfSingle646432
  | .w64, .w64, .w64 => sourceHalfSingle646464

/-- The original selected FP32/FP16/FP32 source, with only duplicate tile-index
arithmetic replaced by its already computed offset_m/offset_n registers. -/
def kernel (abi : SourceABI) (p : KernelArgs) : ComputeKernel :=
  sourceFace abi.startsWidth abi.lengthsWidth abi.adaptersWidth
    .fp32 p.input p.weight p.output p.starts p.lengths p.adapters
    p.columns p.inner p.inputRowStride p.inputInnerStride p.adapterStride
    p.weightColumnStride p.weightInnerStride p.outputRowStride p.outputColumnStride
    p.slice p.batch (p.inner % 16 == 0) abi.addInputs Bool.true

end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated.IndexReuse

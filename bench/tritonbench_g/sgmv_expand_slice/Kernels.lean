import bench.tritonbench_g.sgmv_expand_slice.Proofs.Support.SourceTypes
import VeriTile.Triton.DSL.SignedSingleArgumentRange
namespace VeriTile.Bench.Repairs.SgmvExpandSlice.Pasted
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 40000
def callSignature (input_dtype : FloatDType) (weight_format : LowFormat) (output_format : ScalarFormat)
    (starts_width lengths_width adapters_width : MetadataWidth) : CallSignature where
  pointers := [("lora_ptr",match weight_format with | .half => .fp16 | .bfloat => .bf16),
    ("out_ptr",match output_format with | .half => .fp16 | .bfloat => .bf16 | .single => .fp32),
    ("b_seq_start_loc",match starts_width with | .w32 => .i32 | .w64 => .i64),
    ("seq_lens",match lengths_width with | .w32 => .i32 | .w64 => .i64),
    ("lora_indices",match adapters_width with | .w32 => .i32 | .w64 => .i64)]
  floatingPointers := [("input_ptr",input_dtype)]
  scalars := [("N",.integer),("K",.integer),("xm_stride",.integer),("xk_stride",.integer),("l0_stride",.integer),("lora_k_stride",.integer),("lora_n_stride",.integer),("cm_stride",.integer),("cn_stride",.integer),("slice_offset",.integer),("batch",.integer)]
  constexpr := ["BLOCK_M","BLOCK_N","BLOCK_K","EVEN_K","ADD_INPUTS","CAST_TYPE"]
  readPorts := ["seq_lens","lora_indices","b_seq_start_loc","input_ptr","lora_ptr","out_ptr"]
  writePorts := ["out_ptr"]

def originalKernel (input_dtype : FloatDType) (weight_format : LowFormat) (output_format : ScalarFormat)
    (starts_width lengths_width adapters_width : MetadataWidth)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (BLOCK_M BLOCK_N BLOCK_K : Nat) (EVEN_K ADD_INPUTS CAST_TYPE : Bool) : ComputeKernel :=
  with_triton_call (callSignature input_dtype weight_format output_format starts_width lengths_width adapters_width)
    call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride,slice_offset := slice_offset,batch := batch] in triton {
    pid = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.cast(batch, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    xm_stride = tl.cast(xm_stride, tl.int64)
    xk_stride = tl.cast(xk_stride, tl.int64)
    l0_stride = tl.cast(l0_stride, tl.int64)
    lora_k_stride = tl.cast(lora_k_stride, tl.int64)
    lora_n_stride = tl.cast(lora_n_stride, tl.int64)
    cm_stride = tl.cast(cm_stride, tl.int64)
    cn_stride = tl.cast(cn_stride, tl.int64)
    slice_offset = tl.cast(slice_offset, tl.int64)
    cta_n_num = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    if M <= 0:
        return
    if pid_m * BLOCK_M >= M:
        return
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return
    cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
    offset_m = tl.arange(0, BLOCK_M).to(tl.int64) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N).to(tl.int64) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64)
    ram = offset_m % M
    rbn = offset_n % N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range((K + BLOCK_K - 1) // BLOCK_K):
        global_k = k * BLOCK_K + offset_k
        safe_k = tl.where(global_k < K, global_k, 0)
        a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride +
                 safe_k[None, :] * xk_stride)
        b_ptr = (lora_ptr + l0_stride * lora_index +
                 safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=global_k[None, :] < K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=global_k[:, None] < K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator = tl.dot(tiled_a, tiled_b, accumulator)
    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    offset_cm = cur_seq_start + tl.arange(0, BLOCK_M).to(tl.int64) + pid_m * BLOCK_M
    offset_cn = tl.arange(0, BLOCK_N).to(tl.int64) + pid_n * BLOCK_N + slice_offset
    safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, 0)
    safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
    c_ptr = (out_ptr + safe_cm[:, None] * cm_stride +
             safe_cn[None, :] * cn_stride)
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)
  }

def candidateKernel (input_dtype : FloatDType) (weight_format : LowFormat) (output_format : ScalarFormat)
    (starts_width lengths_width adapters_width : MetadataWidth)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (BLOCK_M BLOCK_N BLOCK_K : Nat) (EVEN_K ADD_INPUTS CAST_TYPE : Bool) : ComputeKernel :=
  with_triton_call (callSignature input_dtype weight_format output_format starts_width lengths_width adapters_width)
    call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride,slice_offset := slice_offset,batch := batch] in triton {
    pid = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.cast(batch, tl.int64)
    N = tl.cast(N, tl.int64)
    K = tl.cast(K, tl.int64)
    xm_stride = tl.cast(xm_stride, tl.int64)
    xk_stride = tl.cast(xk_stride, tl.int64)
    l0_stride = tl.cast(l0_stride, tl.int64)
    lora_k_stride = tl.cast(lora_k_stride, tl.int64)
    lora_n_stride = tl.cast(lora_n_stride, tl.int64)
    cm_stride = tl.cast(cm_stride, tl.int64)
    cn_stride = tl.cast(cn_stride, tl.int64)
    slice_offset = tl.cast(slice_offset, tl.int64)
    cta_n_num = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    if M <= 0:
        return
    if pid_m * BLOCK_M >= M:
        return
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return
    cur_seq_start = tl.load(b_seq_start_loc + cur_batch).to(tl.int64)
    offset_m = tl.arange(0, BLOCK_M).to(tl.int64) + pid_m * BLOCK_M
    offset_n = tl.arange(0, BLOCK_N).to(tl.int64) + pid_n * BLOCK_N
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64)
    ram = offset_m % M
    rbn = offset_n % N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range((K + BLOCK_K - 1) // BLOCK_K):
        global_k = k * BLOCK_K + offset_k
        safe_k = tl.where(global_k < K, global_k, 0)
        a_ptr = (input_ptr + cur_seq_start * xm_stride + ram[:, None] * xm_stride +
                 safe_k[None, :] * xk_stride)
        b_ptr = (lora_ptr + l0_stride * lora_index +
                 safe_k[:, None] * lora_n_stride + rbn[None, :] * lora_k_stride)
        if EVEN_K:
            tiled_a = tl.load(a_ptr)
            tiled_b = tl.load(b_ptr)
        else:
            tiled_a = tl.load(a_ptr,
                              mask=global_k[None, :] < K,
                              other=0)
            tiled_b = tl.load(b_ptr,
                              mask=global_k[:, None] < K,
                              other=0)
        if CAST_TYPE:
            tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
        accumulator = tl.dot(tiled_a, tiled_b, accumulator)
    tiled_c = accumulator.to(lora_ptr.dtype.element_ty)
    offset_cm = cur_seq_start + offset_m
    offset_cn = offset_n + slice_offset
    safe_cm = tl.where(offset_cm < cur_seq_start + M, offset_cm, 0)
    safe_cn = tl.where(offset_cn < slice_offset + N, offset_cn, slice_offset)
    c_ptr = (out_ptr + safe_cm[:, None] * cm_stride +
             safe_cn[None, :] * cn_stride)
    M = tl.load(seq_lens + cur_batch).to(tl.int64)
    c_mask = (offset_cm[:, None] < cur_seq_start + M) & (offset_cn[None, :] < slice_offset + N)
    if ADD_INPUTS:
        tiled_out = tl.load(c_ptr, mask=c_mask)
        tiled_c += tiled_out
    tl.store(c_ptr, tiled_c, mask=c_mask)
  }

end
end VeriTile.Bench.Repairs.SgmvExpandSlice.Pasted

namespace VeriTile.Bench.Repairs.SgmvExpandSlice
open VeriTile.Triton
def sourceHalfSingle323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w32 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w32 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w32 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w32 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w64 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w64 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w64 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfSingle646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .single .w64 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w32 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w32 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w32 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w32 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w64 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w64 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w64 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfHalf646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .half .w64 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w32 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w32 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w32 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w32 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w64 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w64 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w64 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceHalfBfloat646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .half .bfloat .w64 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w32 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w32 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w32 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w32 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w64 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w64 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w64 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatSingle646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .single .w64 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w32 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w32 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w32 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w32 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w64 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w64 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w64 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatHalf646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .half .w64 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat323232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w32 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat323264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w32 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat326432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w32 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat326464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w32 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat643232 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w64 .w32 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat643264 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w64 .w32 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat646432 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w64 .w64 .w32
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceBfloatBfloat646464 (dtype : FloatDType)
    (input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch : Nat) (evenK addInputs castType : Bool) : ComputeKernel :=
  Pasted.originalKernel dtype .bfloat .bfloat .w64 .w64 .w64
    input_ptr lora_ptr out_ptr b_seq_start_loc seq_lens lora_indices
    N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset batch
    32 32 16 evenK addInputs castType

def sourceFace (weight : LowFormat) (output : ScalarFormat)
    (starts lengths adapters : MetadataWidth) :=
  match weight, output, starts, lengths, adapters with
  | .half, .single, .w32, .w32, .w32 => sourceHalfSingle323232
  | .half, .single, .w32, .w32, .w64 => sourceHalfSingle323264
  | .half, .single, .w32, .w64, .w32 => sourceHalfSingle326432
  | .half, .single, .w32, .w64, .w64 => sourceHalfSingle326464
  | .half, .single, .w64, .w32, .w32 => sourceHalfSingle643232
  | .half, .single, .w64, .w32, .w64 => sourceHalfSingle643264
  | .half, .single, .w64, .w64, .w32 => sourceHalfSingle646432
  | .half, .single, .w64, .w64, .w64 => sourceHalfSingle646464
  | .half, .half, .w32, .w32, .w32 => sourceHalfHalf323232
  | .half, .half, .w32, .w32, .w64 => sourceHalfHalf323264
  | .half, .half, .w32, .w64, .w32 => sourceHalfHalf326432
  | .half, .half, .w32, .w64, .w64 => sourceHalfHalf326464
  | .half, .half, .w64, .w32, .w32 => sourceHalfHalf643232
  | .half, .half, .w64, .w32, .w64 => sourceHalfHalf643264
  | .half, .half, .w64, .w64, .w32 => sourceHalfHalf646432
  | .half, .half, .w64, .w64, .w64 => sourceHalfHalf646464
  | .half, .bfloat, .w32, .w32, .w32 => sourceHalfBfloat323232
  | .half, .bfloat, .w32, .w32, .w64 => sourceHalfBfloat323264
  | .half, .bfloat, .w32, .w64, .w32 => sourceHalfBfloat326432
  | .half, .bfloat, .w32, .w64, .w64 => sourceHalfBfloat326464
  | .half, .bfloat, .w64, .w32, .w32 => sourceHalfBfloat643232
  | .half, .bfloat, .w64, .w32, .w64 => sourceHalfBfloat643264
  | .half, .bfloat, .w64, .w64, .w32 => sourceHalfBfloat646432
  | .half, .bfloat, .w64, .w64, .w64 => sourceHalfBfloat646464
  | .bfloat, .single, .w32, .w32, .w32 => sourceBfloatSingle323232
  | .bfloat, .single, .w32, .w32, .w64 => sourceBfloatSingle323264
  | .bfloat, .single, .w32, .w64, .w32 => sourceBfloatSingle326432
  | .bfloat, .single, .w32, .w64, .w64 => sourceBfloatSingle326464
  | .bfloat, .single, .w64, .w32, .w32 => sourceBfloatSingle643232
  | .bfloat, .single, .w64, .w32, .w64 => sourceBfloatSingle643264
  | .bfloat, .single, .w64, .w64, .w32 => sourceBfloatSingle646432
  | .bfloat, .single, .w64, .w64, .w64 => sourceBfloatSingle646464
  | .bfloat, .half, .w32, .w32, .w32 => sourceBfloatHalf323232
  | .bfloat, .half, .w32, .w32, .w64 => sourceBfloatHalf323264
  | .bfloat, .half, .w32, .w64, .w32 => sourceBfloatHalf326432
  | .bfloat, .half, .w32, .w64, .w64 => sourceBfloatHalf326464
  | .bfloat, .half, .w64, .w32, .w32 => sourceBfloatHalf643232
  | .bfloat, .half, .w64, .w32, .w64 => sourceBfloatHalf643264
  | .bfloat, .half, .w64, .w64, .w32 => sourceBfloatHalf646432
  | .bfloat, .half, .w64, .w64, .w64 => sourceBfloatHalf646464
  | .bfloat, .bfloat, .w32, .w32, .w32 => sourceBfloatBfloat323232
  | .bfloat, .bfloat, .w32, .w32, .w64 => sourceBfloatBfloat323264
  | .bfloat, .bfloat, .w32, .w64, .w32 => sourceBfloatBfloat326432
  | .bfloat, .bfloat, .w32, .w64, .w64 => sourceBfloatBfloat326464
  | .bfloat, .bfloat, .w64, .w32, .w32 => sourceBfloatBfloat643232
  | .bfloat, .bfloat, .w64, .w32, .w64 => sourceBfloatBfloat643264
  | .bfloat, .bfloat, .w64, .w64, .w32 => sourceBfloatBfloat646432
  | .bfloat, .bfloat, .w64, .w64, .w64 => sourceBfloatBfloat646464

/-- Complete source selection. EVEN_K and CAST_TYPE are derived from the actual
host dimensions and formats. The mandatory low weight formats are retained. -/
def kernel (abi : SourceABI) (p : KernelArgs) : ComputeKernel :=
  sourceFace abi.weight abi.output abi.startsWidth abi.lengthsWidth abi.adaptersWidth
    abi.input.dtype p.input p.weight p.output p.starts p.lengths p.adapters
    p.columns p.inner p.inputRowStride p.inputInnerStride p.adapterStride
    p.weightColumnStride p.weightInnerStride p.outputRowStride p.outputColumnStride
    p.slice p.batch (p.inner % 16 == 0) abi.addInputs (abi.input != abi.weight.scalar)

def primaryHalf : SourceABI :=
  ⟨.single,.half,.single,.w32,.w32,.w32,Bool.false⟩
def primaryBfloat : SourceABI :=
  ⟨.single,.bfloat,.single,.w32,.w32,.w32,Bool.false⟩

end VeriTile.Bench.Repairs.SgmvExpandSlice

namespace VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated
abbrev source := VeriTile.Bench.Repairs.SgmvExpandSlice.kernel
end VeriTile.Bench.Repairs.SgmvExpandSlice.ByteSeparated

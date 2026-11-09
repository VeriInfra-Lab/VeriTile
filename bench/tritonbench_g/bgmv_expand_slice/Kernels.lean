import bench.tritonbench_g.bgmv_expand_slice.Proofs.Support.Definitions
import VeriTile.Triton.DSL.PythonFullConstructors
namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Pasted
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 24000
inductive IndexStorage where | i32 | i64
  deriving DecidableEq

def callSignature (index_dtype : IndexStorage) : CallSignature where
  pointers := [("input_ptr",.fp32),("lora_ptr",.fp16),("out_ptr",.fp32),
    ("lora_indices",match index_dtype with | .i32 => .i32 | .i64 => .i64)]
  scalars := [("N",.integer),("K",.integer),("xm_stride",.integer),("xk_stride",.integer),("l0_stride",.integer),("lora_k_stride",.integer),("lora_n_stride",.integer),("cm_stride",.integer),("cn_stride",.integer),("slice_offset",.integer)]
  constexpr := ["BLOCK_N","BLOCK_K","SPLIT_N","EVEN_K","ADD_INPUTS","CAST_TYPE"]
  readPorts := ["input_ptr","lora_ptr","out_ptr","lora_indices"]
  writePorts := ["out_ptr"]

def originalKernel (index_dtype : IndexStorage) (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset : Nat) (BLOCK_N BLOCK_K SPLIT_N : Nat) (EVEN_K ADD_INPUTS CAST_TYPE : Bool) : ComputeKernel :=
  with_triton_call (callSignature index_dtype) call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride,slice_offset := slice_offset] in triton {
    """
    GroupGEMV, introducing SPLIT_N can improve large hidden_size's performance
    """
    pid_sn = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64)
    offset_n = tl.arange(0, BLOCK_N).to(tl.int64)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride +
                          offset_k * xk_stride)
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )
    split_n_length = tl.cdiv(tl.cast(N, tl.int64), tl.cast(SPLIT_N, tl.int64))
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    b_ptr = (lora_ptr + l0_stride * lora_index +
             pid_sn * split_n_length * lora_k_stride)
    c_ptr = (out_ptr + cur_batch * cm_stride + pid_sn * split_n_length +
             slice_offset * cn_stride)

    for n in range(tl.cast(0, tl.int64), split_n_length, BLOCK_N):
        current_n = n + offset_n
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :]
                                                              < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr + current_n[:, None] * lora_k_stride +
            offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)
        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)
  }

def candidateKernel (index_dtype : IndexStorage) (input_ptr lora_ptr out_ptr lora_indices : RegionName)
    (N K xm_stride xk_stride l0_stride lora_k_stride lora_n_stride cm_stride cn_stride slice_offset : Nat) (BLOCK_N BLOCK_K SPLIT_N : Nat) (EVEN_K ADD_INPUTS CAST_TYPE : Bool) : ComputeKernel :=
  with_triton_call (callSignature index_dtype) call_args [N := N,K := K,xm_stride := xm_stride,xk_stride := xk_stride,l0_stride := l0_stride,lora_k_stride := lora_k_stride,lora_n_stride := lora_n_stride,cm_stride := cm_stride,cn_stride := cn_stride,slice_offset := slice_offset] in triton {
    """
    GroupGEMV, introducing SPLIT_N can improve large hidden_size's performance
    """
    pid_sn = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64)
    offset_n = tl.arange(0, BLOCK_N).to(tl.int64)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride +
                          offset_k * xk_stride)
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )
    split_n_length = tl.cdiv(tl.cast(N, tl.int64), tl.cast(SPLIT_N, tl.int64))
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    b_ptr = (lora_ptr + l0_stride * lora_index +
             pid_sn * split_n_length * lora_k_stride)
    c_ptr = (out_ptr + cur_batch * cm_stride + pid_sn * split_n_length +
             slice_offset * cn_stride)

    for n in range(tl.cast(0, tl.int64), split_n_length, BLOCK_N):
        current_n = n + offset_n
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :]
                                                              < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr + current_n[:, None] * lora_k_stride +
            offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tiled_out + tl.sum(tiled_a * tiled_b, 1)
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)
        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)
  }

end
end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Pasted

namespace VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Kernels
open VeriTile.Triton
def source (h : Call) : ComputeKernel :=
  Pasted.originalKernel .i64 "X" "W" "Y" "Indices"
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.slice
    256 h.BK h.N h.even h.add Bool.true

def outputFirst (h : Call) : ComputeKernel :=
  Pasted.candidateKernel .i64 "X" "W" "Y" "Indices"
    h.N h.K h.K 1 (h.N*h.K) h.K 1 h.columns 1 h.slice
    256 h.BK h.N h.even h.add Bool.true

end VeriTile.Bench.Repairs.BgmvExpandSlicePhysicalTileChecked.Kernels

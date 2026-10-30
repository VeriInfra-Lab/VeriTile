import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Kernels

/-! Existing unit-split candidate: only cdiv(N,SPLIT_N) is specialized to 1.
The wrapper already supplies SPLIT_N=N and skips N=0. -/
namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated
open VeriTile.Triton
set_option maxRecDepth 40000
set_option maxHeartbeats 8000000

def unitSplitSignatureInt64 : CallSignature where
  pointers := [("input_ptr",.fp32),("lora_ptr",.fp16),("out_ptr",.fp32),("lora_indices",.i64)]
  scalars := [("N",.i32),("K",.i32),("xm_stride",.i32),("xk_stride",.i32),("l0_stride",.i32),("lora_k_stride",.i32),("lora_n_stride",.i32),("cm_stride",.i32),("cn_stride",.i32)]
  constexpr := ["BLOCK_N","BLOCK_K","SPLIT_N","EVEN_K","ADD_INPUTS","CAST_TYPE"]
  readPorts := ["input_ptr","lora_ptr","out_ptr","lora_indices"]
  writePorts := ["out_ptr"]

def unitSplitSignatureInt32 : CallSignature where
  pointers := [("input_ptr",.fp32),("lora_ptr",.fp16),("out_ptr",.fp32),("lora_indices",.i32)]
  scalars := [("N",.i32),("K",.i32),("xm_stride",.i32),("xk_stride",.i32),("l0_stride",.i32),("lora_k_stride",.i32),("lora_n_stride",.i32),("cm_stride",.i32),("cn_stride",.i32)]
  constexpr := ["BLOCK_N","BLOCK_K","SPLIT_N","EVEN_K","ADD_INPUTS","CAST_TYPE"]
  readPorts := ["input_ptr","lora_ptr","out_ptr","lora_indices"]
  writePorts := ["out_ptr"]

def unitSplitSource (p : Branches.Parameters) : ComputeKernel :=
  let input_ptr := p.input
  let lora_ptr := p.weight
  let out_ptr := p.output
  let N := p.outputSize
  let K := p.inner
  let xm_stride := p.inputBatchStride
  let xk_stride := p.inputInnerStride
  let l0_stride := p.loraStride
  let lora_k_stride := p.weightRowStride
  let lora_n_stride := p.weightInnerStride
  let cm_stride := p.outputBatchStride
  let cn_stride := p.outputStride
  let BLOCK_N := p.blockRows
  let BLOCK_K := p.blockInner
  let SPLIT_N := p.splits
  let EVEN_K := p.evenK
  let ADD_INPUTS := p.addInputs
  let CAST_TYPE := p.castType
  let lora_indices : RegionName := p.indices.cast
  with_triton_call unitSplitSignatureInt64 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
    """
    GroupGEMV, additionally, introducing SPLIT_N can improve large hidden_size's
    performance
    """
    pid_sn = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride +
                          offset_k * xk_stride, )  # [BLOCK_K]
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )  # [BLOCK_K]
    # N must be divisible by SPLIT_N
    split_n_length = 1
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    # sliding  to  next row-block
    b_ptr = (lora_ptr + l0_stride * lora_index +
             pid_sn * split_n_length * lora_k_stride)
    c_ptr = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length
    for n in range(0, split_n_length, BLOCK_N):
        current_n = n + offset_n
        current_n_c = tl.max_contiguous(current_n, BLOCK_N)
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :]
                                                              < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr + current_n_c[:, None] * lora_k_stride +
            offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )  # [BLOCK_N,BLOCK_K]
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)

        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)
  }

def unitSplitSourceInt32 (p : Branches.Parameters) : ComputeKernel :=
  let input_ptr := p.input
  let lora_ptr := p.weight
  let out_ptr := p.output
  let N := p.outputSize
  let K := p.inner
  let xm_stride := p.inputBatchStride
  let xk_stride := p.inputInnerStride
  let l0_stride := p.loraStride
  let lora_k_stride := p.weightRowStride
  let lora_n_stride := p.weightInnerStride
  let cm_stride := p.outputBatchStride
  let cn_stride := p.outputStride
  let BLOCK_N := p.blockRows
  let BLOCK_K := p.blockInner
  let SPLIT_N := p.splits
  let EVEN_K := p.evenK
  let ADD_INPUTS := p.addInputs
  let CAST_TYPE := p.castType
  let lora_indices : RegionName := p.indices.cast
  with_triton_call unitSplitSignatureInt32 call_args [N := N, K := K, xm_stride := xm_stride, xk_stride := xk_stride, l0_stride := l0_stride, lora_k_stride := lora_k_stride, lora_n_stride := lora_n_stride, cm_stride := cm_stride, cn_stride := cn_stride] in
  triton {
    """
    GroupGEMV, additionally, introducing SPLIT_N can improve large hidden_size's
    performance
    """
    pid_sn = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride +
                          offset_k * xk_stride, )  # [BLOCK_K]
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )  # [BLOCK_K]
    # N must be divisible by SPLIT_N
    split_n_length = 1
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    # sliding  to  next row-block
    b_ptr = (lora_ptr + l0_stride * lora_index +
             pid_sn * split_n_length * lora_k_stride)
    c_ptr = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length
    for n in range(0, split_n_length, BLOCK_N):
        current_n = n + offset_n
        current_n_c = tl.max_contiguous(current_n, BLOCK_N)
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :]
                                                              < K)
        c_mask = current_n < split_n_length
        tiled_b = tl.load(
            b_ptr + current_n_c[:, None] * lora_k_stride +
            offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )  # [BLOCK_N,BLOCK_K]
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)

        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)
  }


end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated

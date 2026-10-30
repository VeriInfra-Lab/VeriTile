"""Authored from the existing output-first Lean candidate.

Exchange only the final optional FP32 addition operands. All original pointer
calculations, early return, source casts, half reduction and stores remain.
"""
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel_output_first(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    slice_offset,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
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

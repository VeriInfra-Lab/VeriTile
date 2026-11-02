"""Python realization of the retained scalar-left FP32 multiplication candidate.

Only the final scaling operands are exchanged; all signed indices, source
control flow, masks, stores and atomics are copied from the repaired kernel.
"""
import triton
import triton.language as tl


@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    scaling,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_sk = tl.program_id(axis=0).to(tl.int64)
    cur_batch = tl.program_id(axis=1).to(tl.int64)
    lora_index = tl.load(lora_indices + cur_batch).to(tl.int64)
    if lora_index == -1:
        return

    offset_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offset_k = tl.arange(0, BLOCK_K).to(tl.int64) + pid_sk * BLOCK_K
    a_ptr = input_ptr + cur_batch * xm_stride
    b_ptr = lora_ptr + l0_stride * lora_index
    accumulator = tl.zeros((BLOCK_N, ), dtype=tl.float32)
    for k in range(tl.cast(0, tl.int64), tl.cast(K, tl.int64), BLOCK_K * SPLIT_K):
        current_k = k + offset_k
        current_k_c = tl.max_contiguous(current_k, BLOCK_K)
        tiled_a = tl.load(
            a_ptr + current_k_c,
            mask=current_k < K,
            other=0.0,
        )
        b_ptr_mask = (offset_n[:, None] < N) & (current_k[None, :] < K)

        tiled_b = tl.load(
            b_ptr + offset_n[:, None] * lora_k_stride +
            current_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )

        accumulator += tl.sum(tiled_a * tiled_b, 1)
    accumulator = scaling * accumulator
    offset_cn = tl.arange(0, BLOCK_N).to(tl.int64)
    c_ptr = out_ptr + cur_batch * cm_stride + offset_cn * cn_stride
    c_mask = offset_cn < N
    if SPLIT_K == 1:
        tl.store(c_ptr, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptr, accumulator, mask=c_mask)

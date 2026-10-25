"""NEW Python counterparts of the three existing Lean index/mask candidates.
The selected original file is unchanged. Only the documented integer product
or bound test differs; no numerical reduction transformation is introduced.
"""
import triton
import triton.language as tl

@triton.jit
def max_stage1_commuted(
    inp: tl.tensor,
    mid: tl.tensor,
    M: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offset = BLOCK_SIZE * pid + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, max_val)

@triton.jit
def max_stage2_complement(mid: tl.tensor, out: tl.tensor, mid_size: tl.int32, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID).to(tl.int64)
    mid_ptrs = mid + offset
    mask = not (mid_size <= offset)
    mid_val = tl.load(mid_ptrs, mask=mask, other=-float("inf"))
    max_val = tl.max(mid_val)
    tl.store(out, max_val)

@triton.jit
def max_dimension_commuted(
    inp: tl.tensor,
    out_value: tl.tensor,
    out_index: tl.tensor,
    M: tl.int32,
    N: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # set offset
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    m_offset = BLOCK_M * pid_m + tl.arange(0, BLOCK_M).to(tl.int64)
    n_offset = tl.arange(0, BLOCK_N).to(tl.int64)
    offset = m_offset[:, None] * tl.cast(N, tl.int64) * tl.cast(K, tl.int64) + n_offset[None, :] * tl.cast(K, tl.int64) + pid_k
    offset_index = m_offset * tl.cast(K, tl.int64) + pid_k
    # set mask
    mask1 = m_offset < M
    mask = m_offset[:, None] < M and n_offset[None, :] < N
    inp_ptrs = inp + offset
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    result_value, result_index = tl.max(inp_vals, axis=1, return_indices=True)

    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index

    tl.store(out_value_ptrs, result_value, mask=mask1)
    tl.store(out_index_ptrs, result_index, mask=mask1)


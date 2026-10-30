"""Newly authored Python counterpart of the existing Lean row-index candidate.

The original remains in destindex_copy_kv1_repaired.py. Only the source-row
integer product is commuted; masks and floating payload operations are retained.
"""
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_kv_index_commuted(
    K,
    Dest_loc,
    Out,
    stride_k_bs,
    stride_k_h,
    stride_k_d,
    stride_o_bs,
    stride_o_h,
    stride_o_d,
    head_num,
    head_dim,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0)
    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    dest_index = tl.load(Dest_loc + cur_index)

    k_ptrs = K + stride_k_bs * cur_index + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(o_ptrs, k, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    return

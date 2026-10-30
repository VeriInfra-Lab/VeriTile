"""Authored Python counterpart of the pre-existing Lean address candidate.

The original selected JIT is unchanged. This file translates its existing
integer-product commutation; it is not a recovered historical source.
"""
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv_commuted(K, Dest_loc, Out, Out_scale, stride_k_bs, stride_k_h, stride_k_d, stride_dest, stride_o_bs, stride_o_h, stride_o_d, stride_os_bs, stride_os_h, head_num, BLOCK_DMODEL: tl.constexpr, BLOCK_HEAD: tl.constexpr):
    cur_index = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_HEAD).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest)
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + offs_d[None, :] * stride_k_d, mask=offs_h[:, None] < head_num, other=0.0)
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(tl.float16)[:, None]
    denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
    q_src_data = (src_data / denominator).to(tl.int8)
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs, q_src_data, mask=offs_h[:, None] < head_num)
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)

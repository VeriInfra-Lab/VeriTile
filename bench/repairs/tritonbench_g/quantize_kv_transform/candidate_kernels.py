"""Authored Python counterpart of the existing early-address Lean candidate.

The selected original JIT is unchanged. Only the two pure output address
assignments move before floating quantization. This is not a historical source.
"""
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv_address_early(K, Dest_loc, Out, Out_scale, stride_k_bs, stride_k_h, stride_k_d, stride_dest, stride_o_bs, stride_o_h, stride_o_d, stride_os_bs, stride_os_h, head_num, head_dim, BLOCK_DMODEL: tl.constexpr, BLOCK_HEAD: tl.constexpr):
    cur_index = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_HEAD).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest).to(tl.int64)
    src_data = tl.load(K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h + stride_k_d * offs_d[None, :], mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(Out_scale.dtype.element_ty)[:, None]
    denominator = tl.where(maximum[:, None] == 0.0, 1.0, data_scale.to(tl.float32))
    q_src_data = (src_data / denominator).to(tl.int8)
    tl.store(o_ptrs, q_src_data, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)

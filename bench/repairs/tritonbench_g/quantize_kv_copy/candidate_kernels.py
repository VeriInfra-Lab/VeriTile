"""Authored Python counterpart of the existing early-address Lean candidate.

The selected original JIT is unchanged. Only the two pure output address
assignments move before floating quantization. This is not a historical source.
"""
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv_address_early(K, Dest_loc, Out, Out_scale, stride_k_bs, stride_k_h, stride_k_g, stride_k_d, stride_o_bs, stride_o_h, stride_o_g, stride_o_d, stride_os_bs, stride_os_h, stride_os_g, stride_dest, group_size, BLOCK_GROUP_NUM: tl.constexpr, BLOCK_GROUP_DIM: tl.constexpr):
    cur_index = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_g = tl.arange(0, BLOCK_GROUP_NUM).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_GROUP_DIM).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest).to(tl.int64)
    src_data = tl.load(K + cur_index * stride_k_bs + cur_head * stride_k_h + offs_g[:, None] * stride_k_g + offs_d[None, :] * stride_k_d, mask=offs_g[:, None] < group_size, other=0.0)
    o_ptrs = Out + dest_index * stride_o_bs + cur_head * stride_o_h + offs_g[:, None] * stride_o_g + offs_d[None, :] * stride_o_d
    os_ptrs = Out_scale + dest_index * stride_os_bs + cur_head * stride_os_h + offs_g * stride_os_g
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.0).to(Out_scale.dtype.element_ty)
    denominator = tl.where(maximum == 0.0, 1.0, data_scale)
    q_src_data = (src_data / denominator[:, None]).to(tl.int8)
    tl.store(o_ptrs, q_src_data, mask=offs_g[:, None] < group_size)
    tl.store(os_ptrs, data_scale, mask=offs_g < group_size)

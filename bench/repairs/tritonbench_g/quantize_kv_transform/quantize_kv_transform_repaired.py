"""Logical destination scatter with FP32 input, int8 values and FP32 scales."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_destindex_copy_quantize_kv(
    K, Dest_loc, Out, Out_scale,
    stride_k_bs, stride_k_h, stride_k_d, stride_dest,
    stride_o_bs, stride_o_h, stride_o_d,
    stride_os_bs, stride_os_h,
    head_num, head_dim, BLOCK_DMODEL: tl.constexpr, BLOCK_HEAD: tl.constexpr,
):
    cur_index = tl.program_id(0).to(tl.int64)
    offs_h = tl.arange(0, BLOCK_HEAD).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    dest_index = tl.load(Dest_loc + cur_index * stride_dest).to(tl.int64)
    src_data = tl.load(
        K + cur_index * stride_k_bs + offs_h[:, None] * stride_k_h
        + stride_k_d * offs_d[None, :],
        mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0,
    )
    abs_data = tl.abs(src_data)
    maximum = tl.max(abs_data, axis=1)
    data_scale = (maximum / 127.).to(Out_scale.dtype.element_ty)[:, None]
    # Preserve the scale=0 representation of a zero row, with defined codes.
    denominator = tl.where(maximum[:, None] == 0., 1., data_scale.to(tl.float32))
    q_src_data = (src_data / denominator).to(tl.int8)
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]
    os_ptrs = Out_scale + dest_index * stride_os_bs + stride_os_h * offs_h[:, None]
    tl.store(o_ptrs, q_src_data, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    tl.store(os_ptrs, data_scale, mask=offs_h[:, None] < head_num)


def destindex_copy_quantize_kv(K, DestLoc, Out, Out_scale):
    tensors = (K, DestLoc, Out, Out_scale)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all arguments must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided tensors required")
    if K.dtype != torch.float32 or DestLoc.dtype != torch.int32 or Out.dtype != torch.int8 or Out_scale.dtype != torch.float32:
        raise TypeError("FP32 input, int32 destinations, int8 codes and FP32 scales required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if any(t.device != K.device for t in tensors):
        raise ValueError("all tensors must use one device")
    if K.ndim != 3 or DestLoc.ndim != 1 or Out.ndim != 3 or Out_scale.ndim != 3:
        raise ValueError("rank 3/1/3/3 required")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views are unsupported")
    count, heads, dims = DestLoc.numel(), K.shape[1], K.shape[2]
    if heads <= 0 or dims <= 0:
        raise ValueError("positive heads and positive feature width required")
    if K.shape[0] < count or Out.shape[1:] != K.shape[1:] or Out_scale.shape != (Out.shape[0], heads, 1):
        raise ValueError("incompatible tensor shapes")
    if count >= (1 << 31):
        raise ValueError("program count must fit signed int32")
    if any(t.numel() >= (1 << 63) for t in tensors):
        raise ValueError("element extents must fit signed int64")
    bh = triton.next_power_of_2(heads)
    bd = triton.next_power_of_2(dims)
    if bh * bd > (1 << 20):
        raise ValueError("padded tile exceeds 2**20 elements")
    if any(t.numel() > 0 and t.data_ptr() % t.element_size() for t in tensors):
        raise ValueError("naturally aligned data pointers required")
    if any(not (a.data_ptr() + a.element_size() * a.numel() <= b.data_ptr() or
                b.data_ptr() + b.element_size() * b.numel() <= a.data_ptr())
           for i, a in enumerate(tensors) for b in tensors[i + 1:]):
        raise ValueError("Tensor data spans must be disjoint")
    destinations = DestLoc.cpu().tolist()
    if any(d < 0 or d >= Out.shape[0] for d in destinations):
        raise ValueError("destination row out of bounds")
    if len(set(destinations)) != count:
        raise ValueError("destination rows must be unique")
    if count == 0:
        return None
    with torch.cuda.device(K.device):
        _fwd_kernel_destindex_copy_quantize_kv[(count,)](
            K, DestLoc, Out, Out_scale, heads * dims, dims, 1, 1,
            heads * dims, dims, 1, heads, 1, heads, dims,
            BLOCK_DMODEL=bd, BLOCK_HEAD=bh, num_warps=1, num_stages=1)
    return None

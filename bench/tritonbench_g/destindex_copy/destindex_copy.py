import torch

import triton
import triton.language as tl


@triton.jit
def _copy_heads(
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

    k_ptrs = K + cur_index * stride_k_bs + stride_k_h * offs_h[:, None] + stride_k_d * offs_d[None, :]
    o_ptrs = Out + dest_index * stride_o_bs + stride_o_h * offs_h[:, None] + stride_o_d * offs_d[None, :]

    k = tl.load(k_ptrs, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim), other=0.0)
    tl.store(o_ptrs, k, mask=(offs_h[:, None] < head_num) & (offs_d[None, :] < head_dim))
    return


def destindex_copy_kv_repaired(KV_nope, KV_rope, DestLoc, O_nope, O_rope):
    payloads = (KV_nope, KV_rope, O_nope, O_rope)
    tensors = payloads + (DestLoc,)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all arguments must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided tensors required")
    if any(t.ndim != 3 for t in payloads) or DestLoc.ndim != 1:
        raise ValueError("rank-three payloads and vector destinations required")
    if any(t.dtype != torch.float32 for t in payloads) or DestLoc.dtype != torch.int32:
        raise TypeError("FP32 payloads and int32 destinations required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if any(t.device != KV_nope.device for t in tensors):
        raise ValueError("all tensors must use one device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views are unsupported")
    count = DestLoc.numel()
    pairs = ((KV_nope, O_nope), (KV_rope, O_rope))
    if any(k.shape[1:] != out.shape[1:] or count > k.shape[0] for k, out in pairs):
        raise ValueError("payload shapes or source row count do not match")
    if any(t.numel() >= (1 << 31) for t in tensors):
        raise ValueError("tensor extents must fit signed int32 indexing")
    blocks = [(triton.next_power_of_2(k.shape[1]), triton.next_power_of_2(k.shape[2]))
              for k, _ in pairs]
    if any(bh * bd > (1 << 20) for bh, bd in blocks):
        raise ValueError("padded Triton tile exceeds 2**20 elements")
    if any(t.numel() > 0 and t.data_ptr() % 4 != 0 for t in tensors):
        raise ValueError("nonempty data pointers must be 4-byte aligned")
    reads = (KV_nope, KV_rope, DestLoc)
    if any(not (out.data_ptr() + 4 * out.numel() <= t.data_ptr() or
                t.data_ptr() + 4 * t.numel() <= out.data_ptr())
           for out, other in ((O_nope, O_rope), (O_rope, O_nope)) for t in reads + (other,)):
        raise ValueError("outputs must be disjoint from reads and each other")
    destinations = DestLoc.cpu().tolist()
    if any(d < 0 or d >= O_nope.shape[0] or d >= O_rope.shape[0] for d in destinations):
        raise ValueError("destination row out of bounds")
    if len(set(destinations)) != count:
        raise ValueError("destination rows must be unique")
    if count == 0:
        return None
    with torch.cuda.device(KV_nope.device):
        for (k, out), (bh, bd) in zip(pairs, blocks):
            heads, dims = k.shape[1], k.shape[2]
            if heads > 0 and dims > 0:
                _copy_heads[(count,)](k, DestLoc, out, heads * dims, dims, 1,
                    heads * dims, dims, 1, heads, dims, BLOCK_DMODEL=bd,
                    BLOCK_HEAD=bh, num_warps=1, num_stages=1)
    return None

"""Direct FP32 paged K/V copy with the original kernel and squeeze views."""
import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kvcache_seqlen1_kernel(
    K,
    V,
    KCache,
    VCache,
    BLOCK_TABLES,
    context_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_kcb,
    stride_kch,
    stride_kcsplit_x,
    stride_kcs,
    stride_kcd,
    stride_vcb,
    stride_vch,
    stride_vcs,
    stride_vcd,
    stride_bts,
    stride_btb,
    block_size,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size

    range_x = tl.arange(0, KCACHE_X)
    offsets_dmodel_x_partition = tl.arange(0, KCACHE_X)

    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        offsets_k = cur_seq_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel_x_partition * stride_kd
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel_x_partition * stride_vd
        v = tl.load(V + offsets_v)

        offsets_kcache = (
            block_id * stride_kcb
            + cur_kv_head_idx * stride_kch
            + split_x * stride_kcsplit_x
            + offsets_in_last_block * stride_kcs
            + range_x
        )
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = (
            block_id * stride_vcb
            + cur_kv_head_idx * stride_vch
            + offsets_in_last_block * stride_vcs
            + offsets_dmodel_x_partition * stride_vcd
        )
        tl.store(VCache + offsets_vcache, v)
    return


@triton.jit
def _copy_to_kvcache_seqlen1_kernel_commuted(
    K,
    V,
    KCache,
    VCache,
    BLOCK_TABLES,
    context_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_vt,
    stride_vh,
    stride_vd,
    stride_kcb,
    stride_kch,
    stride_kcsplit_x,
    stride_kcs,
    stride_kcd,
    stride_vcb,
    stride_vch,
    stride_vcs,
    stride_vcd,
    stride_bts,
    stride_btb,
    block_size,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    cur_seq_idx = tl.program_id(0)
    cur_kv_head_idx = tl.program_id(1)

    past_kv_seq_len = tl.load(context_lengths + cur_seq_idx) - 1
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb)
    offsets_in_last_block = past_kv_seq_len % block_size

    range_x = tl.arange(0, KCACHE_X)
    offsets_dmodel_x_partition = tl.arange(0, KCACHE_X)

    for split_x in tl.static_range(HEAD_DIM // KCACHE_X):
        offsets_dmodel_x_partition = tl.arange(split_x * KCACHE_X, (split_x + 1) * KCACHE_X)
        offsets_k = cur_kv_head_idx * stride_kh + cur_seq_idx * stride_kt + offsets_dmodel_x_partition * stride_kd
        k = tl.load(K + offsets_k)
        offsets_v = cur_seq_idx * stride_vt + cur_kv_head_idx * stride_vh + offsets_dmodel_x_partition * stride_vd
        v = tl.load(V + offsets_v)

        offsets_kcache = (
            block_id * stride_kcb
            + cur_kv_head_idx * stride_kch
            + split_x * stride_kcsplit_x
            + offsets_in_last_block * stride_kcs
            + range_x
        )
        tl.store(KCache + offsets_kcache, k)
        offsets_vcache = (
            block_id * stride_vcb
            + cur_kv_head_idx * stride_vch
            + offsets_in_last_block * stride_vcs
            + offsets_dmodel_x_partition * stride_vcd
        )
        tl.store(VCache + offsets_vcache, v)
    return


def _checked_copy(k, v, k_cache, v_cache, kv_lengths, block_tables, kernel,
                  use_new_kcache_layout=False):
    tensors = (k, v, k_cache, v_cache, kv_lengths, block_tables)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all arguments must be tensors")
    if type(use_new_kcache_layout) is not bool:
        raise TypeError("cache layout selector must be Boolean")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided tensors required")
    if any(t.dtype != torch.float32 for t in tensors[:4]) or any(t.dtype != torch.int32 for t in tensors[4:]):
        raise TypeError("FP32 payloads and int32 metadata required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if any(t.device != k.device for t in tensors):
        raise ValueError("one CUDA device required")
    if k.ndim not in (3, 4) or v.ndim not in (3, 4) or (k.ndim == 4 and k.shape[1] != 1) or (v.ndim == 4 and v.shape[1] != 1):
        raise ValueError("input shapes must be [N,H,D] or [N,1,H,D]")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views are unsupported")
    k = k.squeeze(1) if k.ndim == 4 else k
    v = v.squeeze(1) if v.ndim == 4 else v
    count, heads, dims = k.shape
    if (v.shape != k.shape or v_cache.ndim != 4 or v_cache.shape[1] != heads or v_cache.shape[3] != dims
            or kv_lengths.ndim != 1 or kv_lengths.shape[0] != count
            or block_tables.ndim != 2 or block_tables.shape[0] != count):
        raise ValueError("incompatible value-cache or metadata shapes")
    blocks, _, block_size, _ = v_cache.shape
    if use_new_kcache_layout:
        if (k_cache.ndim != 5 or k_cache.shape[0] != blocks or k_cache.shape[1] != heads
                or k_cache.shape[3] != block_size or k_cache.shape[2]*k_cache.shape[4] != dims):
            raise ValueError("split cache shape must be [B,H,D/X,S,X]")
        width, splits = k_cache.shape[4], k_cache.shape[2]
    else:
        if k_cache.shape != v_cache.shape:
            raise ValueError("legacy cache shapes must match")
        width, splits = dims, 1
    if heads <= 0 or dims <= 0 or block_size <= 0 or width <= 0 or width & (width-1):
        raise ValueError("positive geometry and power-of-two split width required")
    if count >= (1 << 31) or heads >= (1 << 16) or count*heads >= (1 << 31):
        raise ValueError("grid exceeds signed launcher limits")
    if any(t.numel() >= (1 << 31) for t in tensors) or any(d >= (1 << 31) for d in (dims, block_size, block_tables.shape[1])):
        raise ValueError("signed int32 source arithmetic required")
    if width > (1 << 20):
        raise ValueError("split tile exceeds 2**20 elements")
    if any(t.numel() > 0 and t.data_ptr() % 4 for t in tensors):
        raise ValueError("four-byte alignment required")
    if any(not (a.data_ptr()+4*a.numel() <= b.data_ptr() or b.data_ptr()+4*b.numel() <= a.data_ptr())
           for i,a in enumerate(tensors) for b in tensors[i+1:]):
        raise ValueError("tensor data spans must be disjoint")
    lengths, tables = kv_lengths.cpu().tolist(), block_tables.cpu().tolist()
    destinations = []
    for seq, length in enumerate(lengths):
        if length < 1 or length > block_size*block_tables.shape[1]:
            raise ValueError("context length outside block table")
        column, slot = divmod(length-1, block_size)
        block = tables[seq][column]
        if block < 0 or block >= blocks:
            raise ValueError("block ID outside cache")
        destinations.append((block,slot))
    if len(set(destinations)) != count:
        raise ValueError("cache destinations must be unique")
    if count == 0:
        return None
    split_stride = block_size*width if use_new_kcache_layout else 0
    with torch.cuda.device(k.device):
        kernel[(count, heads)](
            k,v,k_cache,v_cache,block_tables,kv_lengths,
            heads*dims,dims,1,heads*dims,dims,1,
            heads*block_size*dims,block_size*dims,split_stride,width,1,
            heads*block_size*dims,block_size*dims,dims,1,
            block_tables.shape[1],1,block_size,
            HEAD_DIM=dims,KCACHE_X=width,num_warps=8 if dims>128 else 4)
    return None


def copy_kv_to_blocked_cache_repaired(k, v, k_cache, v_cache, kv_lengths, block_tables,
                                      use_new_kcache_layout=False):
    return _checked_copy(k, v, k_cache, v_cache, kv_lengths, block_tables,
                         _copy_to_kvcache_seqlen1_kernel, use_new_kcache_layout)


def copy_kv_to_blocked_cache_commuted(k, v, k_cache, v_cache, kv_lengths, block_tables,
                                      use_new_kcache_layout=False):
    return _checked_copy(k, v, k_cache, v_cache, kv_lengths, block_tables,
                         _copy_to_kvcache_seqlen1_kernel_commuted, use_new_kcache_layout)

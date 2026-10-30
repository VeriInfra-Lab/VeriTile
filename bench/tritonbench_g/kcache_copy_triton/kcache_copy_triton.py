"""Direct FP32 paged copy with original reshape and signed-control repair."""
import torch
import triton
import triton.language as tl

@triton.jit
def _copy_to_kcache_seqlen_n_kernel_repaired(
    K,  # K or V
    KCache,  # [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    BLOCK_TABLES,
    seq_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_kcb,
    stride_kch,
    stride_kcsplit_x,
    stride_kcs,
    stride_kcx,
    stride_bts,
    stride_btb,
    block_size,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    # `n_tokens` is used to specify the number of tokens to copy for each sequence
    # When n_tokens > 1, tokens from different sequences are packed into the first dimension of the grid,
    #   `seq_lengths` must be the lengths of sequences counting the number of tokens to copy
    #   E.g. if n_tokens = 5, seq_lengths = [12, 15], then the already-copied position ids are [0-6, 0-9]
    #   for the two sequences, respectively. And the position ids to be copied are [7-11, 9-14].
    # When n_tokens = 1, consider token idx as the sequence idx, since it's only used during regular decoding stage
    cur_token_idx = tl.program_id(0).to(tl.int64)
    cur_seq_idx = cur_token_idx // n_tokens
    # `cur_token_shift` is only valid and functional when `n_tokens` > 1
    cur_token_shift = cur_token_idx - (n_tokens * (cur_seq_idx + 1))
    cur_kv_head_idx = tl.program_id(1).to(tl.int64)
    split_x_idx = tl.program_id(2).to(tl.int64)

    past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx).to(tl.int64) + cur_token_shift
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb).to(tl.int64)
    offset_last_block = past_kv_seq_len % block_size
    offsets_dmodel = split_x_idx * KCACHE_X + tl.arange(0, KCACHE_X).to(tl.int64)
    offsets_k = cur_token_idx * stride_kt + cur_kv_head_idx * stride_kh + offsets_dmodel * stride_kd
    k = tl.load(K + offsets_k)
    offsets_kcache = (
        block_id * stride_kcb
        + cur_kv_head_idx * stride_kch
        + split_x_idx * stride_kcsplit_x
        + offset_last_block * stride_kcs
        + tl.arange(0, KCACHE_X).to(tl.int64)
    )
    tl.store(KCache + offsets_kcache, k)
    return


@triton.jit
def _copy_to_kcache_seqlen_n_kernel_commuted(
    K,  # K or V
    KCache,  # [num_blocks, num_kv_heads, head_dim // x, block_size, x]
    BLOCK_TABLES,
    seq_lengths,
    stride_kt,
    stride_kh,
    stride_kd,
    stride_kcb,
    stride_kch,
    stride_kcsplit_x,
    stride_kcs,
    stride_kcx,
    stride_bts,
    stride_btb,
    block_size,
    n_tokens,
    HEAD_DIM: tl.constexpr,
    KCACHE_X: tl.constexpr,
):
    # `n_tokens` is used to specify the number of tokens to copy for each sequence
    # When n_tokens > 1, tokens from different sequences are packed into the first dimension of the grid,
    #   `seq_lengths` must be the lengths of sequences counting the number of tokens to copy
    #   E.g. if n_tokens = 5, seq_lengths = [12, 15], then the already-copied position ids are [0-6, 0-9]
    #   for the two sequences, respectively. And the position ids to be copied are [7-11, 9-14].
    # When n_tokens = 1, consider token idx as the sequence idx, since it's only used during regular decoding stage
    cur_token_idx = tl.program_id(0).to(tl.int64)
    cur_seq_idx = cur_token_idx // n_tokens
    # `cur_token_shift` is only valid and functional when `n_tokens` > 1
    cur_token_shift = cur_token_idx - (n_tokens * (cur_seq_idx + 1))
    cur_kv_head_idx = tl.program_id(1).to(tl.int64)
    split_x_idx = tl.program_id(2).to(tl.int64)

    past_kv_seq_len = tl.load(seq_lengths + cur_seq_idx).to(tl.int64) + cur_token_shift
    last_bt_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + cur_seq_idx * stride_bts
    block_id = tl.load(block_table_ptr + last_bt_block_idx * stride_btb).to(tl.int64)
    offset_last_block = past_kv_seq_len % block_size
    offsets_dmodel = split_x_idx * KCACHE_X + tl.arange(0, KCACHE_X).to(tl.int64)
    offsets_k = cur_kv_head_idx * stride_kh + cur_token_idx * stride_kt + offsets_dmodel * stride_kd
    k = tl.load(K + offsets_k)
    offsets_kcache = (
        block_id * stride_kcb
        + cur_kv_head_idx * stride_kch
        + split_x_idx * stride_kcsplit_x
        + offset_last_block * stride_kcs
        + tl.arange(0, KCACHE_X).to(tl.int64)
    )
    tl.store(KCache + offsets_kcache, k)
    return


def _checked_copy(k, k_cache, kv_lengths, block_tables, kernel,
                  n=1, use_new_kcache_layout=False):
    tensors = (k, k_cache, kv_lengths, block_tables)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all arguments must be tensors")
    if type(n) is not int or n <= 0 or n >= (1 << 31):
        raise ValueError("n must be a positive signed-int32 Python integer")
    if type(use_new_kcache_layout) is not bool:
        raise TypeError("cache layout selector must be Boolean")
    if any(t.layout != torch.strided for t in tensors):
        raise TypeError("strided tensors required")
    if k.dtype != torch.float32 or k_cache.dtype != torch.float32 or kv_lengths.dtype != torch.int32 or block_tables.dtype != torch.int32:
        raise TypeError("FP32 payloads and int32 metadata required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if any(t.device != k.device for t in tensors):
        raise ValueError("one CUDA device required")
    if k.ndim not in (3, 4) or kv_lengths.ndim != 1 or block_tables.ndim != 2:
        raise ValueError("rank 3/4 source, rank 1 lengths and rank 2 table required")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors required")
    if any(t.is_neg() for t in tensors):
        raise ValueError("unresolved negative views are unsupported")
    if k.ndim == 4:
        k = k.reshape(k.shape[0]*k.shape[1], k.shape[2], k.shape[3])
    count, heads, dims = k.shape
    if count % n or kv_lengths.shape[0] != count//n or block_tables.shape[0] != count//n:
        raise ValueError("each sequence must supply n tokens and one metadata row")
    if use_new_kcache_layout:
        if k_cache.ndim != 5 or k_cache.shape[1] != heads or k_cache.shape[2]*k_cache.shape[4] != dims:
            raise ValueError("split cache shape must be [B,H,D/X,S,X]")
        blocks, _, splits, block_size, width = k_cache.shape
    else:
        if k_cache.ndim != 4 or k_cache.shape[1] != heads or k_cache.shape[3] != dims:
            raise ValueError("legacy cache shape must be [B,H,S,D]")
        blocks, _, block_size, width = k_cache.shape
        splits = 1
    if heads <= 0 or dims <= 0 or block_size <= 0 or width <= 0 or width & (width-1):
        raise ValueError("positive geometry and power-of-two split width required")
    if count >= (1 << 31) or heads >= (1 << 16) or splits >= (1 << 16) or count*heads*splits >= (1 << 31):
        raise ValueError("grid exceeds signed launcher limits")
    if any(t.numel() >= (1 << 63) for t in tensors) or block_size >= (1 << 63) or block_tables.shape[1] >= (1 << 63):
        raise ValueError("signed int64 address arithmetic required")
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
        if length < n or length > block_size*block_tables.shape[1]:
            raise ValueError("sequence positions outside block table")
        for u in range(n):
            column, slot = divmod(length-n+u, block_size)
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
        kernel[(count, heads, splits)](
            k,k_cache,block_tables,kv_lengths,heads*dims,dims,1,
            heads*block_size*dims,block_size*dims,split_stride,width,1,
            block_tables.shape[1],1,block_size,n_tokens=n,
            HEAD_DIM=dims,KCACHE_X=width,num_warps=8 if dims>128 else 4)
    return None


def copy_k_to_blocked_cache_repaired(k, k_cache, kv_lengths, block_tables,
                                     n=1, use_new_kcache_layout=False):
    return _checked_copy(k, k_cache, kv_lengths, block_tables,
                         _copy_to_kcache_seqlen_n_kernel_repaired,
                         n, use_new_kcache_layout)


def copy_k_to_blocked_cache_commuted(k, k_cache, kv_lengths, block_tables,
                                    n=1, use_new_kcache_layout=False):
    return _checked_copy(k, k_cache, kv_lengths, block_tables,
                         _copy_to_kcache_seqlen_n_kernel_commuted,
                         n, use_new_kcache_layout)

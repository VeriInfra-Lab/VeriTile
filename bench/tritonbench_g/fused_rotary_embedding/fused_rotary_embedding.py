"""Whole decoding rotary/cache repair for ordinary contiguous FP32 tensors.

The public result is None. Q, K-cache and V-cache are mutated; K and V are
read-only. One token represents one decoding sequence. See README.md for the
normal-completed-call contract and the compiler/numerical boundaries.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def decoding_fused_rotary_embedding_kernel(
    q, k, v, cos, sin, k_cache, v_cache, BLOCK_TABLES, context_lengths,
    x, q_token_stride, q_head_stride, q_dim_stride,
    k_token_stride, k_head_stride, k_dim_stride,
    v_token_stride, v_head_stride, v_dim_stride,
    cos_token_stride, cos_stride, sin_token_stride, sin_stride,
    length_stride, kcb_stride, kch_stride, kcsplit_x_stride, kcs_stride,
    kcd_stride, vcb_stride, vch_stride, vcs_stride, vcd_stride,
    bts_stride, btb_stride, block_size,
    KV_GROUP_NUM: tl.constexpr, HEAD_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_idx = tl.program_id(1).to(tl.int64)
    dim_range = tl.arange(0, BLOCK_DIM).to(tl.int64)
    dim_range0 = tl.arange(0, BLOCK_HALF).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2
    pair_mask = dim_range0 < HEAD_DIM // 2

    off_q = cur_token_idx * q_token_stride + cur_head_idx * q_head_stride
    off_q0 = off_q + dim_range0 * q_dim_stride
    off_q1 = off_q + dim_range1 * q_dim_stride
    loaded_q0 = tl.load(q + off_q0, mask=pair_mask, other=0)
    loaded_q1 = tl.load(q + off_q1, mask=pair_mask, other=0)
    off_cos = cur_token_idx * cos_token_stride + dim_range0 * cos_stride
    off_sin = cur_token_idx * sin_token_stride + dim_range0 * sin_stride
    loaded_cos = tl.load(cos + off_cos, mask=pair_mask, other=0)
    loaded_sin = tl.load(sin + off_sin, mask=pair_mask, other=0)
    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos
    tl.store(q + off_q0, out_q0, mask=pair_mask)
    tl.store(q + off_q1, out_q1, mask=pair_mask)

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        cur_k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k = cur_token_idx * k_token_stride + cur_k_head_idx * k_head_stride
        off_k0 = off_k + dim_range0 * k_dim_stride
        off_k1 = off_k + dim_range1 * k_dim_stride
        loaded_k0 = tl.load(k + off_k0, mask=pair_mask, other=0)
        loaded_k1 = tl.load(k + off_k1, mask=pair_mask, other=0)
        out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
        out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos
        past_kv_seq_len = tl.load(context_lengths + cur_token_idx * length_stride).to(tl.int64) - 1
        last_block_idx = past_kv_seq_len // block_size
        block_ids = tl.load(BLOCK_TABLES + cur_token_idx * bts_stride + last_block_idx * btb_stride).to(tl.int64)
        offsets_in_last_block = past_kv_seq_len % block_size
        offsets_cache_base = block_ids * kcb_stride + cur_k_head_idx * kch_stride
        k_range0 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range0 // x) * kcsplit_x_stride
                    + (dim_range0 % x) * kcd_stride)
        k_range1 = (offsets_cache_base + offsets_in_last_block * kcs_stride
                    + (dim_range1 // x) * kcsplit_x_stride
                    + (dim_range1 % x) * kcd_stride)
        tl.store(k_cache + k_range0, out_k0, mask=pair_mask)
        tl.store(k_cache + k_range1, out_k1, mask=pair_mask)
        off_v = (cur_token_idx * v_token_stride + cur_k_head_idx * v_head_stride
                 + dim_range * v_dim_stride)
        loaded_v = tl.load(v + off_v, mask=dim_range < HEAD_DIM, other=0)
        v_range = (block_ids * vcb_stride + cur_k_head_idx * vch_stride
                   + offsets_in_last_block * vcs_stride + dim_range * vcd_stride)
        tl.store(v_cache + v_range, loaded_v, mask=dim_range < HEAD_DIM)


def _ordinary_tensor(tensor, dtype, device):
    if type(tensor) is not torch.Tensor:
        raise TypeError("all inputs must be tensors")
    if tensor.layout != torch.strided or tensor.dtype != dtype or tensor.device != device or not tensor.is_cuda:
        raise ValueError("payloads must be CUDA FP32 and metadata CUDA int32 on q.device")
    if tensor.storage_offset() != 0 or tensor.is_conj() or tensor.is_neg():
        raise ValueError("ordinary zero-offset tensors without lazy flags are required")
    if not tensor.is_contiguous():
        raise ValueError("canonical contiguous tensors are required")
    if tensor.numel() > ((1 << 63) - 1) // 4:
        raise ValueError("tensor byte extent exceeds signed64")
    if tensor.numel():
        stride = 1
        for size, actual in zip(reversed(tensor.shape), reversed(tensor.stride())):
            if actual != stride:
                raise ValueError("canonical contiguous strides are required")
            stride *= size


def decoding_fused_rotary_embedding(
    q, k, v, cos, sin, k_cache=None, v_cache=None, block_tables=None,
    kv_lengths=None, use_new_kcache_layout=False,
):
    """Rotate full split-half Q/K and scatter K/V to checked decoding slots.

    q is [T,HQ,D], k/v [T,HK,D], with positive even D and HQ a positive
    multiple of HK. Coefficient rows align with tokens and contain D/2 or
    more columns. Metadata is int32 [T,M] and [T]. Cache targets (block,slot)
    must be distinct across tokens. Both documented K-cache layouts are
    supported. Noncanonical views and other dtypes are outside this repair.
    """
    if type(q) is not torch.Tensor or not q.is_cuda or q.ndim != 3:
        raise ValueError("q must be a CUDA rank-three tensor")
    if type(use_new_kcache_layout) is not bool:
        raise TypeError("use_new_kcache_layout must be bool")
    floats = (q, k, v, cos, sin, k_cache, v_cache)
    for tensor in floats:
        _ordinary_tensor(tensor, torch.float32, q.device)
    for tensor in (block_tables, kv_lengths):
        _ordinary_tensor(tensor, torch.int32, q.device)
    tokens, q_heads, width = q.shape
    if q_heads <= 0 or width <= 0 or width % 2:
        raise ValueError("positive heads and positive even head dimension are required")
    if k.ndim != 3 or v.shape != k.shape or k.shape[0] != tokens or k.shape[2] != width:
        raise ValueError("K and V must have the same [T,HK,D] shape")
    k_heads = k.shape[1]
    if k_heads <= 0 or q_heads % k_heads:
        raise ValueError("HQ must be a positive multiple of HK")
    for coefficient in (cos, sin):
        if coefficient.ndim != 2 or coefficient.shape[0] < tokens or coefficient.shape[1] < width // 2:
            raise ValueError("coefficients need T rows and D/2 columns")
    if v_cache.ndim != 4 or v_cache.shape[1] != k_heads or v_cache.shape[3] != width:
        raise ValueError("V-cache must be [blocks,HK,block_size,D]")
    blocks, _, block_size, _ = v_cache.shape
    if blocks <= 0 or block_size <= 0:
        raise ValueError("positive cache block count and block size are required")
    if use_new_kcache_layout:
        if (k_cache.ndim != 5 or k_cache.shape[0] != blocks or k_cache.shape[1] != k_heads
                or k_cache.shape[3] != block_size or k_cache.shape[4] <= 0
                or k_cache.shape[2] * k_cache.shape[4] != width):
            raise ValueError("K-cache must be [blocks,HK,D/x,block_size,x]")
        x = k_cache.shape[4]
        kcsplit_x_stride, kcs_stride, kcd_stride = k_cache.stride()[-3:]
    else:
        if k_cache.shape != v_cache.shape:
            raise ValueError("ordinary K-cache must have the V-cache shape")
        x = width
        kcsplit_x_stride, kcs_stride, kcd_stride = 0, k_cache.stride(2), k_cache.stride(3)
    if (kv_lengths.ndim != 1 or kv_lengths.shape[0] != tokens
            or block_tables.ndim != 2 or block_tables.shape[0] != tokens
            or block_tables.shape[1] <= 0):
        raise ValueError("one length and one nonempty block-table row are required per token")
    if tokens == 0:
        return None
    if q_heads > (1 << 31) - 1 or tokens > 65535:
        raise ValueError("CUDA grid axis limit exceeded")
    half_block = triton.next_power_of_2(width // 2)
    full_block = triton.next_power_of_2(width)
    if full_block > 1048576:
        raise ValueError("static rotary tile exceeds Triton maximum 1048576")
    data_pad = width + half_block + full_block
    cache_pad = data_pad * (block_size + 1)
    limits = (q.numel() + data_pad, k.numel() + data_pad, v.numel() + data_pad,
              tokens * cos.shape[1] + half_block, tokens * sin.shape[1] + half_block,
              k_cache.numel() + cache_pad, v_cache.numel() + full_block,
              block_tables.numel(), kv_lengths.numel())
    intervals = []
    for tensor, limit in zip(floats + (block_tables, kv_lengths), limits):
        base = tensor.data_ptr()
        storage = tensor.untyped_storage()
        if base % 4 or base != storage.data_ptr():
            raise ValueError("live tensor pointers must be aligned allocation starts")
        if limit > ((1 << 63) - 1) // 4 or base + 4 * limit > 1 << 64:
            raise ValueError("formed kernel pointers exceed signed-offset or uint64 bounds")
        end = base + storage.nbytes()
        if storage.nbytes() < 4 * tensor.numel() or end > 1 << 64:
            raise ValueError("tensor storage extent is invalid")
        intervals.append((base, end))
    for i, (base, end) in enumerate(intervals):
        for other_base, other_end in intervals[:i]:
            if base < other_end and other_base < end:
                raise ValueError("all nine ordinary allocations must be independent")
    groups = q_heads // k_heads
    num_warps = 16 if width >= 512 else 8 if width >= 256 else 4
    with torch.cuda.device(q.device):
        lengths = kv_lengths.to(device="cpu").tolist()
        tables = block_tables.to(device="cpu").tolist()
        selected = []
        for token, length in enumerate(lengths):
            if length < 1 or length > block_tables.shape[1] * block_size:
                raise ValueError("length must select an existing block-table entry")
            last_block, slot = divmod(length - 1, block_size)
            block = tables[token][last_block]
            if block < 0 or block >= blocks:
                raise ValueError("selected block id is outside the cache")
            selected.append((block, slot))
        if len(set(selected)) != tokens:
            raise ValueError("decoding cache destinations must be distinct")
        decoding_fused_rotary_embedding_kernel[(q_heads, tokens)](
            q, k, v, cos, sin, k_cache, v_cache, block_tables, kv_lengths,
            x, *q.stride(), *k.stride(), *v.stride(),
            *cos.stride(), *sin.stride(), kv_lengths.stride(0),
            k_cache.stride(0), k_cache.stride(1), kcsplit_x_stride, kcs_stride, kcd_stride,
            *v_cache.stride(), *block_tables.stride(), block_size,
            KV_GROUP_NUM=groups, HEAD_DIM=width,
            BLOCK_HALF=half_block,
            BLOCK_DIM=full_block, num_warps=num_warps, enable_fp_fusion=True,
        )
    return None

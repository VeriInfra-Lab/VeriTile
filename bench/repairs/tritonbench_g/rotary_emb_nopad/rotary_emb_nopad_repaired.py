from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def rotary_embedding_kernel(
    q,
    k,
    cos,
    sin,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    sin_token_stride,
    sin_stride,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    KV_GROUP_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,  # token range length
):
    cur_head_idx = tl.program_id(0).to(tl.int64)
    cur_token_block_idx = tl.program_id(1).to(tl.int64)

    tokens_range = cur_token_block_idx * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS).to(tl.int64)
    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2

    off_cos_sin = tokens_range[:, None] * cos_token_stride + dim_range0[None, :] * cos_stride
    loaded_cos = tl.load(cos + off_cos_sin, mask=(tokens_range[:, None] < q_total_tokens) & (dim_range0[None, :] < HEAD_DIM // 2), other=0.0)
    off_sin = tokens_range[:, None] * sin_token_stride + dim_range0[None, :] * sin_stride
    loaded_sin = tl.load(sin + off_sin, mask=(tokens_range[:, None] < q_total_tokens) & (dim_range0[None, :] < HEAD_DIM // 2), other=0.0)

    off_q0 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range0[None, None, :] * head_dim_stride
    )
    off_q1 = (
        tokens_range[:, None, None] * q_token_stride
        + cur_head_idx * q_head_stride
        + dim_range1[None, None, :] * head_dim_stride
    )
    loaded_q0 = tl.load(
        q + off_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
        other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
        other=0.0,
    )
    out_q0 = loaded_q0 * loaded_cos[:, None, :] - loaded_q1 * loaded_sin[:, None, :]
    out_q1 = loaded_q0 * loaded_sin[:, None, :] + loaded_q1 * loaded_cos[:, None, :]

    tl.store(
        q + off_q0,
        out_q0,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
    )
    tl.store(
        q + off_q1,
        out_q1,
        mask=((cur_head_idx < Q_HEAD_NUM) & (tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2)),
    )

    handle_kv = cur_head_idx % KV_GROUP_NUM == 0
    if handle_kv:
        k_head_idx = cur_head_idx // KV_GROUP_NUM
        off_k0 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range0[None, None, :] * head_dim_stride
        )
        off_k1 = (
            tokens_range[:, None, None] * k_token_stride
            + k_head_idx * k_head_stride
            + dim_range1[None, None, :] * head_dim_stride
        )
        loaded_k0 = tl.load(
            k + off_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
            other=0.0,
        )
        loaded_k1 = tl.load(
            k + off_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
            other=0.0,
        )
        out_k0 = loaded_k0 * loaded_cos[:, None, :] - loaded_k1 * loaded_sin[:, None, :]
        out_k1 = loaded_k0 * loaded_sin[:, None, :] + loaded_k1 * loaded_cos[:, None, :]
        tl.store(
            k + off_k0,
            out_k0,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
        )
        tl.store(
            k + off_k1,
            out_k1,
            mask=(tokens_range[:, None, None] < q_total_tokens) & (dim_range0[None, None, :] < HEAD_DIM // 2),
        )


@triton.jit
def fused_rotary_embedding_kernel_v2(
    q,
    k,
    cos,
    sin,
    kv_cache,
    BLOCK_TABLES,
    context_lengths,
    q_token_stride,
    q_head_stride,
    k_token_stride,
    k_head_stride,
    head_dim_stride,
    cos_token_stride,
    cos_stride,
    sin_token_stride,
    sin_stride,
    cacheb_stride,
    cacheh_stride,
    cachebs_stride,
    cached_stride,
    bts_stride,
    btb_stride,
    block_size,
    q_total_tokens,
    Q_HEAD_NUM: tl.constexpr,
    K_HEAD_NUM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_PAIRS: tl.constexpr,
):
    block_head_index = tl.program_id(0).to(tl.int64)
    if block_head_index >= Q_HEAD_NUM:
        return
    block_token_index = tl.program_id(1).to(tl.int64)

    dim_range0 = tl.arange(0, BLOCK_PAIRS).to(tl.int64)
    dim_range1 = dim_range0 + HEAD_DIM // 2

    off_q0 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range0 * head_dim_stride
    off_q1 = block_token_index * q_token_stride + block_head_index * q_head_stride + dim_range1 * head_dim_stride
    off_k0 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range0 * head_dim_stride
    off_k1 = block_token_index * k_token_stride + block_head_index * k_head_stride + dim_range1 * head_dim_stride

    loaded_q0 = tl.load(
        q + off_q0, mask=dim_range0 < HEAD_DIM // 2, other=0.0,
    )
    loaded_q1 = tl.load(
        q + off_q1, mask=dim_range0 < HEAD_DIM // 2, other=0.0,
    )

    loaded_k0 = tl.load(
        k + off_k0, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2), other=0.0,
    )

    loaded_k1 = tl.load(
        k + off_k1, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2), other=0.0,
    )

    off_cos_sin = block_token_index * cos_token_stride + dim_range0 * cos_stride

    loaded_cos = tl.load(cos + off_cos_sin, mask=(block_token_index < q_total_tokens) & (dim_range0 < HEAD_DIM // 2), other=0.0)
    off_sin = block_token_index * sin_token_stride + dim_range0 * sin_stride
    loaded_sin = tl.load(sin + off_sin, mask=(block_token_index < q_total_tokens) & (dim_range0 < HEAD_DIM // 2), other=0.0)

    out_q0 = loaded_q0 * loaded_cos - loaded_q1 * loaded_sin
    out_q1 = loaded_q0 * loaded_sin + loaded_q1 * loaded_cos

    out_k0 = loaded_k0 * loaded_cos - loaded_k1 * loaded_sin
    out_k1 = loaded_k0 * loaded_sin + loaded_k1 * loaded_cos  # total_tokens, head_num, head_dim

    past_kv_seq_len = tl.load(context_lengths + block_token_index) - 1

    last_block_idx = past_kv_seq_len // block_size
    block_table_ptr = BLOCK_TABLES + block_token_index * bts_stride
    block_ids = tl.load(block_table_ptr + last_block_idx * btb_stride, mask=(block_token_index < q_total_tokens))
    offsets_in_last_block = (past_kv_seq_len % block_size) * cachebs_stride

    kv_range0 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range0 * cached_stride
    )
    kv_range1 = (
        block_ids * cacheb_stride
        + block_head_index * cacheh_stride
        + offsets_in_last_block
        + dim_range1 * cached_stride
    )

    tl.store(
        kv_cache + kv_range0,
        out_k0, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2),
    )
    tl.store(
        kv_cache + kv_range1,
        out_k1, mask=(block_head_index < K_HEAD_NUM) & (dim_range0 < HEAD_DIM // 2),
    )

    # concat
    tl.store(
        q + off_q0,
        out_q0, mask=dim_range0 < HEAD_DIM // 2,
    )
    tl.store(
        q + off_q1,
        out_q1, mask=dim_range0 < HEAD_DIM // 2,
    )



def _span(tensor):
    if tensor.numel() == 0:
        return None
    storage = tensor.untyped_storage()
    base, size = storage.data_ptr(), storage.nbytes()
    unit = tensor.element_size()
    if (tensor.data_ptr() != base or base % unit or unit*tensor.numel() > size
            or base+size > 2**64):
        raise ValueError("aligned contained canonical backing storage is required")
    return base,base+size


def _check_inputs(q, k, cos, sin, k_cache, block_tables, kv_lengths):
    payloads = [q,k,cos,sin] + ([] if k_cache is None else [k_cache])
    if any(not isinstance(t,torch.Tensor) for t in payloads):
        raise TypeError("payload arguments must be tensors")
    for t in payloads:
        if (t.layout != torch.strided or t.dtype != torch.float32 or not t.is_cuda
                or t.device != q.device or not t.is_contiguous() or t.storage_offset() != 0
                or t.is_neg() or t.is_conj()):
            raise ValueError("canonical FP32 CUDA payloads on one device are required")
    if q.ndim != 3 or k.ndim != 3 or cos.ndim != 2 or sin.ndim != 2:
        raise ValueError("Q/K need rank three and coefficients rank two")
    tokens,hq,width = q.shape
    hk = k.shape[1]
    if hq <= 0 or hk <= 0 or hq % hk or width % 2:
        raise ValueError("positive GQA-compatible heads and an even head width are required")
    if k.shape[0] != tokens or k.shape[2] != width:
        raise ValueError("Q/K token counts and head widths must agree")
    half = width//2
    if any(t.shape[0] < tokens or t.shape[1] < half for t in (cos,sin)):
        raise ValueError("coefficient tables must contain every active token/pair")
    tensors = list(payloads)
    if k_cache is not None:
        if k_cache.ndim != 4 or k_cache.shape[1] != hk or k_cache.shape[3] != width or k_cache.shape[2] <= 0:
            raise ValueError("cache must have shape [blocks,K_heads,positive_block_size,width]")
        for t in (block_tables,kv_lengths):
            if (not isinstance(t,torch.Tensor) or t.layout != torch.strided or t.dtype != torch.int64
                    or not t.is_cuda or t.device != q.device or not t.is_contiguous()
                    or t.storage_offset() != 0 or t.is_neg() or t.is_conj()):
                raise ValueError("cache metadata must be contiguous CUDA int64 on the Q device")
        if block_tables.ndim != 2 or block_tables.shape[0] != tokens or kv_lengths.shape != (tokens,):
            raise ValueError("cache metadata needs one row/length per input token")
        if tokens and kv_lengths.stride(0) != 1:
            raise ValueError("the original length load requires unit stride")
        tensors.extend((block_tables,kv_lengths))
    spans = [span for t in tensors if (span := _span(t)) is not None]
    if any(a < d and c < b for i,(a,b) in enumerate(spans) for c,d in spans[i+1:]):
        raise ValueError("all caller backing allocations must be physically separate")
    if tokens == 0 or width == 0:
        return
    if torch.cuda.current_device() != q.device.index:
        raise ValueError("the original launch requires the Q CUDA device to be current")
    for t in tensors:
        stride = 1
        for n,s in zip(reversed(t.shape),reversed(t.stride())):
            if s != stride:
                raise ValueError("exact canonical launch strides are required")
            stride *= max(1,n)
    if any(t.numel() >= 2**60 for t in tensors):
        raise ValueError("source byte extents must fit signed int64")
    pairs = triton.next_power_of_2(half)
    padded_tokens = triton.cdiv(tokens,4)*4
    data_last = [(padded_tokens-1)*t.stride(0)+(hq-1)*width+half+pairs-1 for t in (q,k)]
    coefficient_last = [(padded_tokens-1)*t.stride(0)+pairs-1 for t in (cos,sin)]
    bounds = list(zip((q,k,cos,sin),data_last+coefficient_last))
    if k_cache is not None:
        blocks,_,block_size,_ = k_cache.shape
        lengths = kv_lengths.to(device="cpu").tolist()
        tables = block_tables.to(device="cpu").tolist()
        selected = []
        for token,length in enumerate(lengths):
            if length <= 0:
                raise ValueError("cache lengths must be positive")
            page,slot = divmod(length-1,block_size)
            if page >= block_tables.shape[1]:
                raise ValueError("selected cache page is outside the block table")
            block = tables[token][page]
            if block < 0 or block >= blocks:
                raise ValueError("selected cache block is outside the allocation")
            selected.append((block,slot))
        if len(set(selected)) != tokens:
            raise ValueError("cache token destinations must be distinct")
        last = ((blocks-1)*k_cache.stride(0)+(hq-1)*k_cache.stride(1)
                +(block_size-1)*width+half+pairs-1)
        bounds.append((k_cache,last))
    if any(4*last+4 >= 2**63 or t.data_ptr()+4*last+4 > 2**64 for t,last in bounds):
        raise ValueError("padded signed64 offsets and formed pointers must fit")

def rotary_embedding(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
    kv_lengths: Optional[torch.Tensor] = None,
):
    """
    Args:
        q: query tensor, [total_tokens, head_num, head_dim]
        k: key tensor, [total_tokens, kv_head_num, head_dim]
        cos: cosine for rotary embedding, [max_position_len, head_dim]
        sin: sine for rotary embedding, [max_position_len, head_dim]
        k_cache (torch.Tensor):  Blocked key cache. [num_blocks, num_kv_heads, block_size, head_dim]
        kv_lengths, Past key/value sequence lengths plus current sequence length for each sequence. [bsz]
        block_tables: Block tables for each sequence. [bsz, max_blocks_per_sequence]
    """
    _check_inputs(q, k, cos, sin, k_cache, block_tables, kv_lengths)
    q_total_tokens, q_head_num, head_dim = q.shape
    if q_total_tokens == 0 or head_dim == 0:
        return None
    block_pairs = triton.next_power_of_2(head_dim // 2)
    assert q.size(0) == k.size(0)
    BLOCK_TOKENS = 4

    if head_dim >= 512:
        num_warps = 16
    elif head_dim >= 256:
        num_warps = 8
    else:
        num_warps = 4

    k_head_num = k.size(1)
    q_token_stride, q_head_stride, head_dim_stride = q.stride()
    k_token_stride, k_head_stride, _ = k.stride()
    cos_token_stride, cos_stride = cos.stride()

    assert q_head_num % k_head_num == 0
    kv_group_num = q_head_num // k_head_num

    if k_cache == None:
        grid = lambda META: (
            q_head_num,
            triton.cdiv(q_total_tokens, META["BLOCK_TOKENS"]),
        )
        rotary_embedding_kernel[grid](
            q,
            k,
            cos,
            sin,
            q_token_stride,
            q_head_stride,
            k_token_stride,
            k_head_stride,
            head_dim_stride,
            cos_token_stride,
            cos_stride,
            sin.stride(0),
            sin.stride(1),
            q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            KV_GROUP_NUM=kv_group_num,
            HEAD_DIM=head_dim,
            BLOCK_PAIRS=block_pairs,
            BLOCK_TOKENS=BLOCK_TOKENS,
            num_warps=num_warps,
        )
    else:
        grid = (triton.next_power_of_2(q_head_num), q_total_tokens)
        fused_rotary_embedding_kernel_v2[grid](
            q,
            k,
            cos,
            sin,
            k_cache,
            block_tables,
            kv_lengths,
            q_token_stride,
            q_head_stride,
            k_token_stride,
            k_head_stride,
            head_dim_stride,
            cos_token_stride,
            cos_stride,
            sin.stride(0),
            sin.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            block_tables.stride(0),
            block_tables.stride(1),
            k_cache.size(-2),
            q_total_tokens,
            Q_HEAD_NUM=q_head_num,
            K_HEAD_NUM=k_head_num,
            HEAD_DIM=head_dim,
            BLOCK_PAIRS=block_pairs,
            num_warps=num_warps,
        )
    return





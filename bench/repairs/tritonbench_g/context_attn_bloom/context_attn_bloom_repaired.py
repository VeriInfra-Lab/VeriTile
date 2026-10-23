import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(Q, K, V, sm_scale, B_Start_Loc, B_Seqlen, Out, Req_to_tokens, B_req_idx, stride_qbs, stride_qh, stride_qd, stride_kbs, stride_kh, stride_kd, stride_vbs, stride_vh, stride_vd, stride_obs, stride_oh, stride_od, stride_req_to_tokens_b, stride_req_to_tokens_s, kv_group_num, b_prompt_cache_len, head_dim: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    start_m = tl.program_id(2).to(tl.int64)
    cur_kv_head = cur_head // kv_group_num
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch).to(tl.int64)
    block_start_loc = BLOCK_M * start_m
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs + cur_head * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(Q + off_q, mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < head_dim), other=0.0)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum((start_m + 1) * BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_loc = tl.load(Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * (start_n + offs_n), mask=start_n + offs_n < block_end_loc, other=0).to(tl.int64)
        off_k = kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :] < block_end_loc) & (offs_d[:, None] < head_dim), other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale
        qk = tl.where(offs_m[:, None] + prompt_cache_len >= start_n + offs_n[None, :], qk, float('-100000000.0'))
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc_scale = tl.where(offs_m + prompt_cache_len >= start_n, acc_scale, 1.0)
        acc = acc * acc_scale[:, None]
        off_v = kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None] < block_end_loc) & (offs_d[None, :] < head_dim), other=0.0)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        l_i = l_i_new
        m_i = m_i_new
    off_o = (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs + cur_head * stride_oh + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < cur_batch_seq_len) & (offs_d[None, :] < head_dim))
    return

_I32_MAX = (1 << 31) - 1
_I64_WORD_MAX = ((1 << 63) - 1) // 4

def _checked_tensor(x, dtype, device, name):
    if type(x) is not torch.Tensor or x.layout != torch.strided or not x.is_cuda:
        raise TypeError(f"{name} must be an ordinary strided CUDA Tensor")
    if x.dtype != dtype or x.device != device:
        raise TypeError(f"{name} has the wrong dtype or owner device")
    if x.is_neg() or x.is_conj():
        raise ValueError(f"{name} must have resolved negative/conjugate flags")
    if any(s < 0 for s in x.stride()):
        raise ValueError(f"{name} must have nonnegative strides")
    if x.storage_offset() < 0 or x.untyped_storage().nbytes() % 4 or x.untyped_storage().nbytes() >= 1 << 63:
        raise ValueError(f"{name} has an invalid four-byte storage ABI")
    if x.numel() and x.data_ptr() % 4:
        raise ValueError(f"{name} requires a four-byte aligned data pointer")
    if x.numel():
        last = x.storage_offset() + sum((n - 1) * s for n, s in zip(x.shape, x.stride()))
        if last >= x.untyped_storage().nbytes() // 4:
            raise ValueError(f"{name} view exceeds its allocation")


def _formed_bound(x, shape):
    # Includes masked, formed pointers; the bound is checked before the launch.
    bound = x.storage_offset() + sum(max(n - 1, 0) * s for n, s in zip(shape, x.stride()))
    if bound > _I64_WORD_MAX or any(s > _I64_WORD_MAX for s in x.stride()):
        raise ValueError("formed pointer arithmetic exceeds signed INT64 byte range")


@torch.no_grad()
def context_attention_fwd(q, k, v, o, b_req_idx, b_start_loc, b_seq_len,
                          b_prompt_cache_len, max_input_len, req_to_token_indexs):
    """Direct checked FP32 page-gather attention; update o and return None."""
    if type(q) is not torch.Tensor:
        raise TypeError("q must be an ordinary Tensor")
    owner = q.device
    for name, tensor in (("q", q), ("k", k), ("v", v), ("o", o)):
        _checked_tensor(tensor, torch.float32, owner, name)
        if tensor.ndim != 3 or not tensor.is_contiguous():
            raise ValueError("payloads must be contiguous rank-three tensors")
    rows, heads, dimension = q.shape
    cache_rows, kv_heads, cache_dimension = k.shape
    if (o.shape != q.shape or v.shape != k.shape or dimension != cache_dimension or
            not 16 <= dimension <= 256 or heads <= 0 or
            kv_heads <= 0 or heads % kv_heads):
        raise ValueError("invalid payload shapes or grouped-head geometry")
    for tensor in (q, k, v, o):
        if tensor.stride() != (tensor.shape[1] * dimension, dimension, 1):
            raise ValueError("payloads must have canonical contiguous strides")
    vectors = (("b_req_idx", b_req_idx), ("b_start_loc", b_start_loc),
               ("b_seq_len", b_seq_len), ("b_prompt_cache_len", b_prompt_cache_len))
    for name, tensor in vectors:
        _checked_tensor(tensor, torch.int32, owner, name)
        if tensor.ndim != 1 or not tensor.is_contiguous() or tensor.shape != b_seq_len.shape:
            raise ValueError("vector metadata must share one contiguous int32 shape")
    table = req_to_token_indexs
    _checked_tensor(table, torch.int32, owner, "req_to_token_indexs")
    if table.ndim != 2 or not table.is_contiguous() or table.stride() != (table.shape[1], 1):
        raise ValueError("page table must have canonical contiguous int32 rows")
    batch = b_seq_len.numel()
    if type(max_input_len) is not int or not 0 <= max_input_len <= _I32_MAX:
        raise ValueError("max_input_len must be a nonnegative signed-INT32 Python int")
    if batch > _I32_MAX or heads > 65535:
        raise ValueError("head grid exceeds the native CUDA y limit")
    for tensor in (q, o):
        _formed_bound(tensor, (rows + max_input_len + 128, heads, triton.next_power_of_2(dimension)))
    for tensor in (k, v):
        _formed_bound(tensor, (max(cache_rows, 1), kv_heads, triton.next_power_of_2(dimension)))
    for _, tensor in vectors:
        _formed_bound(tensor, tensor.shape)
    _formed_bound(table, (table.shape[0], table.shape[1] + 128))
    if batch == 0:
        return None
    def separate(a, b):
        sa, sb = a.untyped_storage(), b.untyped_storage()
        return (sa.nbytes() == 0 or sb.nbytes() == 0 or
                sa.data_ptr() + sa.nbytes() <= sb.data_ptr() or
                sb.data_ptr() + sb.nbytes() <= sa.data_ptr())
    if any(not separate(o, t) for t in (q, k, v, *(x for _, x in vectors), table)):
        raise ValueError("output backing storage must be disjoint from every read role")
    with torch.cuda.device(owner):
        starts = b_start_loc.cpu().tolist()
        sequences = b_seq_len.cpu().tolist()
        prompts = b_prompt_cache_len.cpu().tolist()
        requests = b_req_idx.cpu().tolist()
        pages = table.cpu().tolist()
        intervals, lengths = [], []
        for start, sequence, prompt, request in zip(starts, sequences, prompts, requests):
            if (start < 0 or prompt < 0 or sequence < prompt or sequence > table.shape[1] or
                    request < 0 or request >= table.shape[0]):
                raise ValueError("invalid sequence, prompt or request metadata")
            length = sequence - prompt
            if length > max_input_len or start + length > rows:
                raise ValueError("insufficient query capacity or max_input_len")
            if any(page < 0 or page >= cache_rows for page in pages[request][:sequence]):
                raise ValueError("page table contains an invalid cache row")
            lengths.append(length)
            if length:
                intervals.append((start, start + length))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("packed output row intervals must not overlap")
        maximum = max(lengths, default=0)
        if maximum == 0:
            return None
        block = 64 if "Tesla" in torch.cuda.get_device_name(owner) else 128
        blocks = triton.cdiv(maximum, block)
        if blocks > 65535:
            raise ValueError("query-block grid exceeds the native CUDA z limit")
        _fwd_kernel[(batch, heads, blocks)](
            q, k, v, 1.0 / (dimension ** 0.5),
            b_start_loc, b_seq_len, o, table, b_req_idx,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(), *table.stride(),
            kv_group_num=heads // kv_heads, b_prompt_cache_len=b_prompt_cache_len,
            head_dim=dimension, BLOCK_DMODEL=triton.next_power_of_2(dimension), BLOCK_M=block, BLOCK_N=block,
            num_warps=4 if dimension <= 64 else 8, num_stages=1,
        )
    return None

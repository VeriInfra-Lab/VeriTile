"""Checked FP32 context attention with the original finite-sentinel computation."""
import torch
import triton
import triton.language as tl

ORIGINAL_SHA256 = "4540e1ffb44e922bb2aa48236b0a43af75a5f185906ffcd52d25a508fb9c1370"
_I32_MAX = (1 << 31) - 1
_I64_WORD_MAX = ((1 << 63) - 1) // 4

@triton.jit
def _fwd_kernel_int8kv(
    Q,
    K,
    V,
    sm_scale,
    Out,
    B_Start_Loc,
    B_Seqlen,
    b_prompt_cache_len,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0).to(tl.int64)
    cur_bh = tl.program_id(1).to(tl.int64)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H

    cur_kv_head = cur_head // kv_group_num
    prompt_cache_len = tl.load(b_prompt_cache_len + cur_batch).to(tl.int64)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch).to(tl.int64)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64) - prompt_cache_len

    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    offs_n = tl.arange(0, BLOCK_N).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL).to(tl.int64)
    offs_m = block_start_loc + tl.arange(0, BLOCK_M).to(tl.int64)
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=offs_m[:, None] < cur_batch_seq_len, other=0.0)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_mask = tl.where(block_start_loc < cur_batch_seq_len, 1, 0)
    block_end_loc = tl.minimum(block_start_loc + BLOCK_M + prompt_cache_len, cur_batch_seq_len + prompt_cache_len)
    # causal mask
    for start_n in range(0, block_mask * block_end_loc, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        # k = tl.load(
        #     k_ptrs + (start_n + offs_n[None, :]) * stride_ks,
        #     mask=(start_n + offs_n[None, :]) < block_end_loc,
        #     other=0,
        # )
        off_k = (
            cur_batch * stride_kb
            + (start_n + offs_n[None, :]) * stride_ks
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(K + off_k, mask=(start_n + offs_n[None, :]) < block_end_loc, other=0.0)

        qk = tl.dot(q, k)
        mask = (offs_m[:, None] + prompt_cache_len) >= (start_n + offs_n[None, :])
        qk = tl.where(mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        # v = tl.load(
        #     v_ptrs + (start_n + offs_n[:, None]) * stride_vs,
        #     mask=(start_n + offs_n[:, None]) < block_end_loc,
        #     other=0.0,
        # )
        off_v = (
            cur_batch * stride_vb
            + (start_n + offs_n[:, None]) * stride_vs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(V + off_v, mask=(start_n + offs_n[:, None]) < block_end_loc, other=0.0)

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)
        # update m_i
        m_i = m_ij

    acc = acc / l_i[:, None]
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < cur_batch_seq_len)


def _checked_tensor(x, dtype, device, name):
    if type(x) is not torch.Tensor or x.layout != torch.strided or not x.is_cuda:
        raise TypeError(f"{name} must be an ordinary strided CUDA Tensor")
    if x.dtype != dtype or x.device != device:
        raise TypeError(f"{name} has the wrong dtype or owner device")
    if x.is_neg() or x.is_conj():
        raise ValueError(f"{name} must have resolved negative/conjugate flags")
    if any(s < 0 for s in x.stride()):
        raise ValueError(f"{name} must have nonnegative strides")
    if x.storage_offset() < 0 or x.untyped_storage().nbytes() % 4:
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
def context_attention_fwd_ppl_int8kv(q, k, v, o, b_start_loc, b_seq_len, max_input_len, b_prompt_cache_len):
    """Mutate supplied o and return None on the documented checked FP32 domain."""
    if type(q) is not torch.Tensor:
        raise TypeError("q must be an ordinary Tensor")
    owner = q.device
    for name, tensor in (("q", q), ("k", k), ("v", v), ("o", o)):
        _checked_tensor(tensor, torch.float32, owner, name)
    for name, tensor in (("b_start_loc", b_start_loc), ("b_seq_len", b_seq_len),
                         ("b_prompt_cache_len", b_prompt_cache_len)):
        _checked_tensor(tensor, torch.int32, owner, name)
        if tensor.ndim != 1 or not tensor.is_contiguous():
            raise ValueError("metadata must have a contiguous vector ABI")
    if q.ndim != 3 or o.shape != q.shape or k.ndim != 4 or v.ndim != 4:
        raise ValueError("expected Q/Out[T,H,D] and independent K/V[B,Hkv,S,D]")
    batch, head, dimension = b_seq_len.numel(), q.shape[1], q.shape[2]
    if b_start_loc.shape != b_seq_len.shape or b_prompt_cache_len.shape != b_seq_len.shape:
        raise ValueError("metadata vector lengths differ")
    if k.shape[0] != batch or v.shape[0] != batch:
        raise ValueError("cache batch dimensions differ from metadata")
    if dimension not in (16, 32, 64, 128, 256) or k.shape[3] != dimension or v.shape[3] != dimension:
        raise ValueError("unsupported head width")
    if head <= 0 or k.shape[1] <= 0 or v.shape[1] != k.shape[1] or head % k.shape[1]:
        raise ValueError("invalid grouped-query head dimensions")
    if not q.is_contiguous() or not o.is_contiguous():
        raise ValueError("query and output must be contiguous")
    # Caches retain the original batch-padded [B,Hkv,S,D] layout, including
    # adjacent K/V head slices of a shared allocation. Inner axes are dense.
    for tensor in (k, v):
        if tensor.numel():
            b, h, n, d = tensor.shape
            s = tensor.stride()
            if (s[3] != 1 or (n > 1 and s[2] != d) or
                    (h > 1 and s[1] != n * d) or
                    (b > 1 and s[0] < h * n * d)):
                raise ValueError("cache inner axes must be dense, with disjoint batch slices")
    if type(max_input_len) is not int or not 0 <= max_input_len <= _I32_MAX:
        raise ValueError("max_input_len must be a nonnegative signed-INT32 Python int")
    if batch * head > 65535:
        raise ValueError("fused batch/head grid exceeds the CUDA y limit")
    # Bounds are independent of the actual device's 64/128 tile choice.
    _formed_bound(q, (q.shape[0] + max_input_len + 128, head, dimension))
    _formed_bound(o, (o.shape[0] + max_input_len + 128, head, dimension))
    for tensor in (k, v):
        _formed_bound(tensor, (batch, tensor.shape[1], tensor.shape[2] + 128, dimension))
    for tensor in (b_start_loc, b_seq_len, b_prompt_cache_len):
        _formed_bound(tensor, tensor.shape)
    if batch == 0:
        return None
    def separate(a, b):
        sa, sb = a.untyped_storage(), b.untyped_storage()
        return (sa.nbytes() == 0 or sb.nbytes() == 0 or
                sa.data_ptr() + sa.nbytes() <= sb.data_ptr() or
                sb.data_ptr() + sb.nbytes() <= sa.data_ptr())
    if any(not separate(o, t) for t in (q, k, v, b_start_loc, b_seq_len, b_prompt_cache_len)):
        raise ValueError("output backing storage must be disjoint from all read roles")
    with torch.cuda.device(owner):
        starts = b_start_loc.cpu().tolist()
        sequences = b_seq_len.cpu().tolist()
        prompts = b_prompt_cache_len.cpu().tolist()
        intervals = []
        for start, sequence, prompt in zip(starts, sequences, prompts):
            if not (0 <= start and 0 <= prompt <= sequence <= min(k.shape[2], v.shape[2])):
                raise ValueError("invalid start/prompt/sequence metadata or cache capacity")
            length = sequence - prompt
            if length > max_input_len or start + length > q.shape[0]:
                raise ValueError("max_input_len or packed query capacity is insufficient")
            if length:
                intervals.append((start, start + length))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("packed output row intervals must not overlap")
        if not intervals:
            return None
        block = 64 if "Tesla" in torch.cuda.get_device_name(owner) else 128
        grid = (triton.cdiv(max_input_len, block), batch * head, 1)
        scale = 1.0 / (dimension ** 0.5) * 1.4426950408889634
        _fwd_kernel_int8kv[grid](
            q, k, v, scale, o, b_start_loc, b_seq_len, b_prompt_cache_len,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2),
            kv_group_num=head // k.shape[1], H=head, BLOCK_DMODEL=dimension,
            BLOCK_M=block, BLOCK_N=block, num_warps=4 if dimension <= 64 else 8,
            num_stages=1,
        )
    return None

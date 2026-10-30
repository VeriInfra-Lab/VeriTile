
import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_kernel_token_att2_affine(
    Prob,
    V,
    Out,
    Req_to_tokens,
    B_req_idx,
    B_Start_Loc,
    B_Seqlen,
    B_Att_Start_Loc,
    B_Att_Seqlen,
    stride_b_req_idx, stride_b_seq_len, stride_b_att_start, stride_b_att_len,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_ph,
    stride_pbs,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    kv_group_num,
    sliding_window,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Triton kernel for computing token attention
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch * stride_b_seq_len)
    cur_batch_start_index = tl.maximum(cur_batch_seq_len - sliding_window, 0)
    cur_batch_in_all_start_index = tl.load(B_Att_Start_Loc + cur_batch * stride_b_att_start)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch * stride_b_req_idx)
    cur_att_seq_len = tl.load(B_Att_Seqlen + cur_batch * stride_b_att_len)

    v_loc_off = (
        cur_batch_req_idx * stride_req_to_tokens_b + (cur_batch_start_index + offs_n) * stride_req_to_tokens_s
    )
    p_offs = (cur_head * stride_ph + cur_batch_in_all_start_index * stride_pbs) + offs_n * stride_pbs
    v_offs = cur_kv_head * stride_vh + offs_d[None, :] * stride_vd

    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    for start_n in range(0, cur_att_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        p_value = tl.load(Prob + p_offs + start_n * stride_pbs, mask=(start_n + offs_n) < cur_att_seq_len, other=0.0)
        v_loc = tl.load(
            Req_to_tokens + v_loc_off + start_n * stride_req_to_tokens_s,
            mask=(start_n + offs_n + cur_batch_start_index) < cur_batch_seq_len,
            other=0.0,
        )
        v_value = tl.load(
            V + v_offs + v_loc[:, None] * stride_vbs,
            mask=(start_n + offs_n[:, None] + cur_batch_start_index) < cur_batch_seq_len,
            other=0.0,
        )
        acc += tl.sum(p_value[:, None] * v_value, 0)

    acc = acc.to(Out.dtype.element_ty)
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


def _checked_arguments(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window):
    metadata = (B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen)
    tensors = (prob, v, out, Req_to_tokens, *metadata)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all buffers must be tensors")
    if prob.ndim != 2 or v.ndim != 3 or out.ndim != 3 or Req_to_tokens.ndim != 2:
        raise ValueError("expected P[Hq,T], V[cache,Hkv,D], Out[B,Hq,D], table[R,M]")
    if any(t.ndim != 1 for t in metadata):
        raise ValueError("batch metadata must be vectors")
    batch, head, dim, kvheads = B_req_idx.numel(), prob.shape[0], v.shape[2], v.shape[1]
    if dim not in (16, 32, 64, 128):
        raise ValueError("supported D is 16, 32, 64 or 128")
    if head <= 0 or kvheads <= 0 or head % kvheads:
        raise ValueError("positive Hq divisible by positive Hkv required")
    if out.shape != (batch, head, dim) or any(t.shape != (batch,) for t in metadata):
        raise ValueError("output or batch metadata shape mismatch")
    if any(t.dtype != torch.float32 for t in (prob, v, out)):
        raise TypeError("FP32 P, V and Out required")
    if any(t.dtype != torch.int32 for t in (Req_to_tokens, *metadata)):
        raise TypeError("signed-int32 request table and metadata required")
    if not prob.is_cuda or any(t.device != prob.device for t in tensors):
        raise ValueError("all buffers must share one CUDA device")
    if any(t.is_neg() or not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors without lazy-negative views required")
    if type(sliding_window) is not int or not 0 <= sliding_window < 2**31:
        raise ValueError("window must be a nonnegative signed-int32 Python integer")
    # Padded source expressions are evaluated before their load masks.
    if (any(t.numel() > 2**31-128 for t in tensors)
        or prob.numel()+128 >= 2**31
        or Req_to_tokens.numel()+prob.shape[1]+128 >= 2**31
        or Req_to_tokens.shape[1]+prob.shape[1]+128 >= 2**31):
        raise ValueError("tensor extents exceed the padded signed-int32 contract")
    requests = B_req_idx.tolist()
    sequences = B_Seqlen.tolist()
    starts, lengths = B_Att_Start_Loc.tolist(), B_Att_Seqlen.tolist()
    for b, (request, sequence, start, length) in enumerate(zip(requests, sequences, starts, lengths)):
        if not 0 <= request < Req_to_tokens.shape[0]:
            raise ValueError("request index is out of bounds")
        if not 0 <= sequence <= Req_to_tokens.shape[1]:
            raise ValueError("sequence length is out of bounds")
        if length < 0 or start < 0 or start+length > prob.shape[1]:
            raise ValueError("probability interval is out of bounds")
        first = max(sequence-sliding_window, 0)
        # Validate the current sequence's cache suffix, including V lanes
        # that can be loaded when the independent P tail mask is false.
        active = Req_to_tokens[request, first:sequence]
        if active.numel() and (active.min().item() < 0 or active.max().item() >= v.shape[0]):
            raise ValueError("cache page index is out of bounds")
    if not batch:
        return False
    if head > 65535:
        raise ValueError("query heads exceed the CUDA grid y limit 65535")
    # A shared typed storage cell has one dtype; same-dtype read aliases remain valid.
    for floating in (prob, v):
        for integer in (Req_to_tokens, *metadata):
            if (floating.numel() and integer.numel()
                and floating.data_ptr() < integer.data_ptr()+4*integer.numel()
                and integer.data_ptr() < floating.data_ptr()+4*floating.numel()):
                raise ValueError("FP32 and int32 input views must not overlap")
    for t in tensors:
        if t.data_ptr()+4*t.numel() > 2**64:
            raise ValueError("buffer exceeds the native pointer address space")
        if t.numel() and t.data_ptr()%4:
            raise ValueError("nonempty buffers require four-byte alignment")
    begin, end = out.data_ptr(), out.data_ptr()+4*out.numel()
    for t in (prob, v, Req_to_tokens, *metadata):
        lo, hi = t.data_ptr(), t.data_ptr()+4*t.numel()
        if t.numel() and lo < end and begin < hi:
            raise ValueError("Out must not overlap any input buffer")
    return True


@torch.no_grad()
def token_att_fwd2_affine(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window):
    """Checked FP32 gathered weighted sum; mutates out and returns None."""
    if not _checked_arguments(prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen, B_Att_Start_Loc, B_Att_Seqlen, sliding_window):
        return
    batch, heads, dim, kvheads = B_req_idx.numel(), prob.shape[0], v.shape[2], v.shape[1]
    with torch.cuda.device(prob.device):
        _fwd_kernel_token_att2_affine[(batch, heads)](
            prob, v, out, Req_to_tokens, B_req_idx, B_Start_Loc, B_Seqlen,
            B_Att_Start_Loc, B_Att_Seqlen,
            1, 1, 1, 1,
            Req_to_tokens.shape[1], 1, prob.shape[1], 1,
            kvheads*dim, dim, 1, heads*dim, dim, 1,
            kv_group_num=heads//kvheads,
            sliding_window=sliding_window,
            BLOCK_DMODEL=dim, BLOCK_N=128, num_warps=4, num_stages=1,
        )


import triton
import triton.language as tl
import torch

@triton.jit
def _fwd_kernel_token_att1_reuse(
    Q, K, sm_scale, B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    Att_Out,
    stride_b_loc_b, stride_b_loc_s,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    att_stride_h, att_stride_bs,
    kv_group_num,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_n = tl.program_id(2)
    
    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)

    cur_batch_start_index = max_input_len - cur_batch_seq_len
    cur_batch_end_index = max_input_len

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd

    block_stard_index = start_n * BLOCK_N
    offs_n = block_stard_index + tl.arange(0, BLOCK_N)
    block_mask = tl.where(block_stard_index < cur_batch_seq_len, 1, 0)

    for start_mark in range(0, block_mask, 1):
        q = tl.load(Q + off_q + start_mark)
        offs_n_new = cur_batch_start_index + offs_n
        k_loc = tl.load(B_Loc + stride_b_loc_b * cur_batch + stride_b_loc_s * offs_n_new, mask=offs_n_new < cur_batch_end_index, other=0)
        off_k = k_loc[:, None] * stride_kbs + cur_kv_head * stride_kh + offs_d[None, :] * stride_kd
        k = tl.load(K + off_k, mask=offs_n_new[:, None] < cur_batch_end_index, other=0.0)
        att_value = tl.sum(q[None, :] * k, 1)
        att_value *= sm_scale
        off_o = cur_head * att_stride_h + (cur_batch_in_all_start_index + offs_n) * att_stride_bs
        tl.store(Att_Out + off_o, att_value, mask=offs_n_new < cur_batch_end_index)
    return

def _checked_arguments(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    tensors = (q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all buffers must be tensors")
    if q.ndim != 3 or k.ndim != 3 or att_out.ndim != 2 or B_Loc.ndim != 2:
        raise ValueError("expected Q[B,Hq,D], K[cache,Hkv,D], Out[Hq,T], Loc[B,M]")
    if B_Start_Loc.ndim != 1 or B_Seqlen.ndim != 1:
        raise ValueError("Start and Length must be vectors")
    batch, heads, dim = q.shape
    pages, kvheads, kdim = k.shape
    if dim != kdim or dim not in (16, 32, 64, 128):
        raise ValueError("matching D in {16,32,64,128} required")
    if heads <= 0 or kvheads <= 0 or heads % kvheads:
        raise ValueError("positive Hq divisible by positive Hkv required")
    if type(max_input_len) is not int or max_input_len < 0:
        raise ValueError("maximum length must be a nonnegative Python integer")
    if B_Loc.shape != (batch, max_input_len) or B_Start_Loc.shape != (batch,) or B_Seqlen.shape != (batch,):
        raise ValueError("metadata shapes must match Q batch and maximum length")
    if att_out.shape[0] != heads:
        raise ValueError("Out must have one row per query head")
    if any(t.dtype != torch.float32 for t in (q, k, att_out)):
        raise TypeError("FP32 Q, K and Out required")
    if any(t.dtype != torch.int32 for t in (B_Loc, B_Start_Loc, B_Seqlen)):
        raise TypeError("signed-int32 metadata required")
    if not q.is_cuda or any(t.device != q.device for t in tensors):
        raise ValueError("all buffers must share one CUDA device")
    if any(t.is_neg() or not t.is_contiguous() for t in tensors):
        raise ValueError("contiguous tensors without lazy-negative views required")
    tokens = att_out.shape[1]
    expected = ((heads*dim, dim, 1), (kvheads*dim, dim, 1), (tokens, 1), (max_input_len, 1), (1,), (1,))
    # Empty calls form no addresses. Their standard Torch strides may differ.
    if batch and max_input_len and any(t.stride() != st for t, st in zip(tensors, expected)):
        raise ValueError("canonical dense tensor strides required for nonempty launch")
    qbs, qh, qd = q.stride(); kbs, kh, kd = k.stride()
    lb, ls = B_Loc.stride(); oh, os = att_out.stride()
    envelope = (batch + heads + kvheads + dim + max_input_len + 32 + pages + tokens
                + q.numel() + k.numel() + B_Loc.numel() + att_out.numel()
                + batch*qbs + heads*qh + dim*qd + pages*kbs + kvheads*kh + dim*kd
                + lb*batch + ls*(max_input_len+32) + heads*oh + (tokens+32)*os)
    if envelope >= 2**31:
        raise ValueError("shape/stride envelope exceeds signed-int32 source arithmetic")
    # Only integer metadata is inspected; floating payloads remain caller-owned.
    starts, lengths = B_Start_Loc.tolist(), B_Seqlen.tolist()
    segments = []
    for b, (start, length) in enumerate(zip(starts, lengths)):
        if not 0 <= length <= max_input_len or not 0 <= start <= tokens-length:
            raise ValueError("invalid sequence length or output segment")
        if length:
            active = B_Loc[b, max_input_len-length:max_input_len]
            if active.min().item() < 0 or active.max().item() >= pages:
                raise ValueError("active cache locations must be in bounds")
            segments.append((start, start+length))
    if any(a[0] < b[1] and b[0] < a[1]
           for i, a in enumerate(segments) for b in segments[i+1:]):
        raise ValueError("active output segments must not overlap")
    if batch == 0 or max_input_len == 0:
        return False
    if heads > 65535 or triton.cdiv(max_input_len, 32) > 65535:
        raise ValueError("CUDA grid y/z exceeds the per-axis limit")
    # A shared typed storage cell has one dtype; same-dtype read aliases remain valid.
    for floating in (q, k):
        for integer in (B_Loc, B_Start_Loc, B_Seqlen):
            if (floating.numel() and integer.numel()
                and floating.data_ptr() < integer.data_ptr()+4*integer.numel()
                and integer.data_ptr() < floating.data_ptr()+4*floating.numel()):
                raise ValueError("FP32 and int32 input views must not overlap")
    for t in tensors:
        if t.data_ptr()+4*t.numel() > 2**64:
            raise ValueError("buffer exceeds the native pointer address space")
        if t.numel() and t.data_ptr() % 4:
            raise ValueError("nonempty buffers require four-byte alignment")
    out_begin, out_end = att_out.data_ptr(), att_out.data_ptr()+att_out.numel()*4
    if att_out.numel():
        for t in (q, k, B_Loc, B_Start_Loc, B_Seqlen):
            begin, end = t.data_ptr(), t.data_ptr()+t.numel()*4
            if t.numel() and begin < out_end and out_begin < end:
                raise ValueError("Out must not overlap any input buffer")
    return True


@torch.no_grad()
def token_att_fwd_reuse(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
    """Checked FP32 gathered QK score stage; mutates att_out and returns None."""
    if not _checked_arguments(q, k, att_out, B_Loc, B_Start_Loc, B_Seqlen, max_input_len):
        return
    dim = q.shape[-1]
    with torch.cuda.device(q.device):
        _fwd_kernel_token_att1_reuse[(q.shape[0], q.shape[1], triton.cdiv(max_input_len, 32))](
            q, k, 1.0/(dim**0.5), B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
            att_out, *B_Loc.stride(), *q.stride(), *k.stride(), *att_out.stride(),
            kv_group_num=q.shape[1]//k.shape[1],
            BLOCK_DMODEL=dim, BLOCK_N=32, num_warps=2, num_stages=1,
        )

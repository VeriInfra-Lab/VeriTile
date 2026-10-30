
import triton
import triton.language as tl
import torch


@triton.jit
def _fwd_kernel(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    stride_logic_h, stride_logic_bs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_b_loc_b, stride_b_loc_s,
    other_kv_index, # Avoid reading NaN data
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_v = cur_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)

        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs, 
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
    
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        n_e_max = tl.maximum(n_e_max, -340282346638528859811704183484516925440.0)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs,
                    mask=(start_n + offs_n < cur_batch_seq_len)[:, None], other=0.0)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    acc = acc / e_sum
    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)
    return


def _check_contents(locations, starts, lengths, batch, tokens, pages, maximum):
    for b in range(batch):
        start, count = int(starts[b]), int(lengths[b])
        if start < 0 or count <= 0 or count > maximum or start + count > tokens:
            raise ValueError("invalid nonempty sequence bounds")
        for k in range(count):
            page = int(locations[b, maximum - count + k])
            if page < 0 or page >= pages:
                raise ValueError("active page index is out of bounds")


@torch.no_grad()
def token_softmax_reducev_fwd(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    # Native decoding layout: V[page, head, channel], page table right aligned.
    tensors = (logics, v, o, b_loc, b_start_loc, b_seq_len)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all six inputs must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("strided tensors required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if logics.ndim != 2 or v.ndim != 3 or o.ndim != 3:
        raise ValueError("expected logits rank2, values/output rank3")
    if b_loc.ndim != 2 or b_start_loc.ndim != 1 or b_seq_len.ndim != 1:
        raise ValueError("expected page table rank2 and metadata rank1")
    batch, heads = b_seq_len.shape[0], logics.shape[0]
    dims = v.shape[2]
    if dims <= 0 or dims & (dims - 1):
        raise ValueError("head dimension must be a positive power of two")
    if type(max_input_len) is not int or max_input_len <= 0:
        raise ValueError("max_input_len must be positive")
    if v.shape[1] != heads or o.shape != (batch, heads, dims):
        raise ValueError("value/output shapes do not match heads and channels")
    if b_start_loc.shape != b_seq_len.shape or b_loc.shape != (batch, max_input_len):
        raise ValueError("page table and metadata shapes do not match")
    if any(t.dtype != torch.float32 for t in (logics, v, o)):
        raise TypeError("FP32 logits, values and output required")
    if any(t.dtype != torch.int32 for t in (b_loc, b_start_loc, b_seq_len)):
        raise TypeError("signed-int32 page and sequence metadata required")
    tensors = (logics, v, o, b_loc, b_start_loc, b_seq_len)
    if any(t.device != logics.device or t.is_neg() or not t.is_contiguous() for t in tensors):
        raise ValueError("ordinary contiguous tensors on one device required")
    if max_input_len > 2**31 - 64 or any(t.numel() > 2**31 - 64 for t in tensors):
        raise ValueError("tensor extents exceed the signed-int32 address contract")
    if batch == 0 or heads == 0:
        return
    if batch >= 2**31 or heads >= 2**16:
        raise ValueError("grid exceeds CUDA launch dimensions")
    if dims * 64 > 2**20:
        raise ValueError("tile exceeds the Triton frontend limit")
    if any(t.numel() and t.data_ptr() % 4 for t in tensors):
        raise ValueError("four-byte pointer alignment required")
    output_storage = o.untyped_storage()
    out_lo = output_storage.data_ptr()
    out_hi = out_lo + 4 * (output_storage.nbytes() // 4)
    for t in (logics, v, b_loc, b_start_loc, b_seq_len):
        storage = t.untyped_storage()
        lo = storage.data_ptr()
        hi = lo + 4 * (storage.nbytes() // 4)
        if not (out_hi <= lo or hi <= out_lo):
            raise ValueError("output storage must be separate from read storage")
    _check_contents(b_loc, b_start_loc, b_seq_len, batch, logics.shape[1],
                    v.shape[0], max_input_len)
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)
    num_warps = 1
    with torch.cuda.device(logics.device):
        _fwd_kernel[grid](
            logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len,
            logics.shape[1], 1,
            heads * dims, dims, 1,
            heads * dims, dims, 1,
            max_input_len, 1,
            other_kv_index,
            BLOCK_DMODEL=v.shape[-1],
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=3
        )
    return





@triton.jit
def _fwd_kernel_early_address(
    Logics, V, Out,
    B_Loc, B_Start_Loc, B_Seqlen, max_input_len,
    stride_logic_h, stride_logic_bs,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_b_loc_b, stride_b_loc_s,
    other_kv_index, # Avoid reading NaN data
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_start_loc = tl.load(B_Start_Loc + cur_batch)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_v = cur_head * stride_vh + offs_d[None, :] * stride_vd
    off_b_loc = cur_batch * stride_b_loc_b + (max_input_len - cur_batch_seq_len) * stride_b_loc_s

    v_ptrs = V + off_v

    e_max = float("-inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, cur_batch_seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v_index = tl.load(B_Loc + off_b_loc + (start_n + offs_n) * stride_b_loc_s, mask=(start_n + offs_n) < cur_batch_seq_len, other=0)

        qk = tl.load(Logics + cur_head * stride_logic_h + (cur_batch_start_loc + start_n + offs_n) * stride_logic_bs, 
                     mask=start_n + offs_n < cur_batch_seq_len, other=float("-inf"))
    
        n_e_max = tl.maximum(tl.max(qk, 0), e_max)
        n_e_max = tl.maximum(n_e_max, -340282346638528859811704183484516925440.0)
        old_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max)
        e_sum = e_sum * old_scale + tl.sum(p, 0)
        v = tl.load(v_ptrs + v_index[:, None] * stride_vbs,
                    mask=(start_n + offs_n < cur_batch_seq_len)[:, None], other=0.0)
        acc = acc * old_scale + tl.sum(p[:, None] * v, 0)
        e_max = n_e_max

    off_o = cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od
    out_ptrs = Out + off_o
    acc = acc / e_sum
    tl.store(out_ptrs, acc)
    return



@torch.no_grad()
def token_softmax_reducev_early_address(logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len, other_kv_index):
    # Native decoding layout: V[page, head, channel], page table right aligned.
    tensors = (logics, v, o, b_loc, b_start_loc, b_seq_len)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all six inputs must be tensors")
    if any(t.layout != torch.strided for t in tensors):
        raise ValueError("strided tensors required")
    if any(not t.is_cuda for t in tensors):
        raise ValueError("CUDA tensors required")
    if logics.ndim != 2 or v.ndim != 3 or o.ndim != 3:
        raise ValueError("expected logits rank2, values/output rank3")
    if b_loc.ndim != 2 or b_start_loc.ndim != 1 or b_seq_len.ndim != 1:
        raise ValueError("expected page table rank2 and metadata rank1")
    batch, heads = b_seq_len.shape[0], logics.shape[0]
    dims = v.shape[2]
    if dims <= 0 or dims & (dims - 1):
        raise ValueError("head dimension must be a positive power of two")
    if type(max_input_len) is not int or max_input_len <= 0:
        raise ValueError("max_input_len must be positive")
    if v.shape[1] != heads or o.shape != (batch, heads, dims):
        raise ValueError("value/output shapes do not match heads and channels")
    if b_start_loc.shape != b_seq_len.shape or b_loc.shape != (batch, max_input_len):
        raise ValueError("page table and metadata shapes do not match")
    if any(t.dtype != torch.float32 for t in (logics, v, o)):
        raise TypeError("FP32 logits, values and output required")
    if any(t.dtype != torch.int32 for t in (b_loc, b_start_loc, b_seq_len)):
        raise TypeError("signed-int32 page and sequence metadata required")
    tensors = (logics, v, o, b_loc, b_start_loc, b_seq_len)
    if any(t.device != logics.device or t.is_neg() or not t.is_contiguous() for t in tensors):
        raise ValueError("ordinary contiguous tensors on one device required")
    if max_input_len > 2**31 - 64 or any(t.numel() > 2**31 - 64 for t in tensors):
        raise ValueError("tensor extents exceed the signed-int32 address contract")
    if batch == 0 or heads == 0:
        return
    if batch >= 2**31 or heads >= 2**16:
        raise ValueError("grid exceeds CUDA launch dimensions")
    if dims * 64 > 2**20:
        raise ValueError("tile exceeds the Triton frontend limit")
    if any(t.numel() and t.data_ptr() % 4 for t in tensors):
        raise ValueError("four-byte pointer alignment required")
    output_storage = o.untyped_storage()
    out_lo = output_storage.data_ptr()
    out_hi = out_lo + 4 * (output_storage.nbytes() // 4)
    for t in (logics, v, b_loc, b_start_loc, b_seq_len):
        storage = t.untyped_storage()
        lo = storage.data_ptr()
        hi = lo + 4 * (storage.nbytes() // 4)
        if not (out_hi <= lo or hi <= out_lo):
            raise ValueError("output storage must be separate from read storage")
    _check_contents(b_loc, b_start_loc, b_seq_len, batch, logics.shape[1],
                    v.shape[0], max_input_len)
    BLOCK = 64
    batch, head = b_seq_len.shape[0], logics.shape[0]
    grid = (batch, head)
    num_warps = 1
    with torch.cuda.device(logics.device):
        _fwd_kernel_early_address[grid](
            logics, v, o, b_loc, b_start_loc, b_seq_len, max_input_len,
            logics.shape[1], 1,
            heads * dims, dims, 1,
            heads * dims, dims, 1,
            max_input_len, 1,
            other_kv_index,
            BLOCK_DMODEL=v.shape[-1],
            BLOCK_N=BLOCK,
            num_warps=num_warps,
            num_stages=3
        )
    return




"""Canonical FP32 stage-2 summary merge; positive active lengths only."""
import torch
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2(B_Seqlen, Mid_O, Mid_O_LogExpSum, Out, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, head_dim, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os, mask=offs_d < head_dim, other=0.0)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(Out + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp, mask=offs_d < head_dim)
    return

@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, Out, block_seq):
    tensors = (mid_out, mid_out_logexpsum, B_Seqlen, Out)
    if any(not isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all buffers must be tensors")
    if any(t.layout != torch.strided or not t.is_cuda for t in tensors):
        raise TypeError("all buffers must be strided CUDA tensors")
    if any(t.dtype != torch.float32 for t in (mid_out, mid_out_logexpsum, Out)) or B_Seqlen.dtype != torch.int32:
        raise TypeError("summary/output buffers must be FP32 and lengths must be int32")
    if any(t.device != mid_out.device for t in tensors):
        raise ValueError("all buffers must share the owner device")
    if any(not t.is_contiguous() or t.storage_offset() != 0 or t.is_neg() or t.is_conj() for t in tensors):
        raise ValueError("buffers must be contiguous with zero offsets and no lazy flags")
    if mid_out.ndim != 4:
        raise ValueError("mid_out must have shape [batch, heads, blocks, dimension]")
    batch, head_num, capacity, head_dim = mid_out.shape
    if head_dim <= 0 or mid_out_logexpsum.shape != (batch, head_num, capacity) or B_Seqlen.shape != (batch,) or Out.shape != (batch, head_num, head_dim):
        raise ValueError("summary, length and output shapes must agree and dimension must be positive")
    if type(block_seq) is not int or not 0 < block_seq <= 2**31 - 1:
        raise ValueError("block_seq must be a positive signed-int32 Python integer")
    BLOCK_DMODEL = triton.next_power_of_2(head_dim)
    if head_dim > BLOCK_DMODEL or BLOCK_DMODEL > 2**30 or batch > 2**31 - 1 or head_num > 65535:
        raise ValueError("dimension and grid must fit the checked native index domain")
    if any(t.numel() + BLOCK_DMODEL > (2**63 - 1) // 4 for t in tensors):
        raise ValueError("formed byte addresses must fit signed 64-bit indexing")
    if batch == 0 or head_num == 0:
        return
    spans = []
    for tensor in tensors:
        if tensor.numel() > 0:
            pointer = tensor.data_ptr()
            if pointer % 4 or pointer + 4 * (tensor.numel() + BLOCK_DMODEL) > 2**64:
                raise ValueError("native pointer alignment or uint64 span rejected")
        storage = tensor.untyped_storage()
        spans.append((storage.data_ptr(), storage.data_ptr() + storage.nbytes()))
    def disjoint(i, j):
        return spans[i][1] <= spans[j][0] or spans[j][1] <= spans[i][0]
    if not all(disjoint(3, j) for j in (0, 1, 2)):
        raise ValueError("output storage must be disjoint from the read-only inputs")
    if not all(disjoint(2, j) for j in (0, 1)):
        raise ValueError("int32 length storage must be disjoint from FP32 summary storage")
    with torch.cuda.device(mid_out.device):
        for b in range(batch):
            length = int(B_Seqlen[b])
            if not 0 < length <= min(capacity * block_seq, 2**31 - 1):
                raise ValueError("each active length must be positive and fit the supplied summary capacity")
        _fwd_kernel_flash_decode_stage2[(batch, head_num)](
            B_Seqlen, mid_out, mid_out_logexpsum, Out,
            head_num * capacity * head_dim, capacity * head_dim, head_dim, 1,
            head_num * capacity, capacity, 1,
            head_num * head_dim, head_dim, 1,
            head_dim,
            BLOCK_SEQ=block_seq, BLOCK_DMODEL=BLOCK_DMODEL,
            num_warps=4, num_stages=2,
        )
    return

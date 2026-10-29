"""Authored from the pre-existing Lean shared-base CSE candidate.

The canonical wrapper binds its stride ABI to H/S/D. Reuse the scalar log-row
base for the vector-row base; all original casts, recurrence and masks remain.
"""
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_flash_decode_stage2_shared_base(B_Seqlen, Mid_O, Mid_O_LogExpSum, O, stride_mid_ob, stride_mid_oh, stride_mid_os, stride_mid_od, stride_mid_o_eb, stride_mid_o_eh, stride_mid_o_es, stride_obs, stride_oh, stride_od, BLOCK_SEQ: tl.constexpr, BLOCK_DMODEL: tl.constexpr):
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_head = tl.program_id(1).to(tl.int64)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch).to(tl.int64)
    block_n_size = tl.where(cur_batch_seq_len <= 0, 0, cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    sum_exp = 0.0
    max_logic = -float('inf')
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    offs_v = offs_logic * stride_mid_os + offs_d
    for block_seq_n in range(0, block_n_size, 1):
        tv = tl.load(Mid_O + offs_v + block_seq_n * stride_mid_os)
        tlogic = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n)
        new_max_logic = tl.maximum(tlogic, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - new_max_logic)
        acc += exp_logic * tv
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic
    tl.store(O + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / sum_exp)
    return

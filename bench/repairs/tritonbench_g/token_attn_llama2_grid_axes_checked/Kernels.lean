import bench.repairs.tritonbench_g.token_attn_llama2.Kernels
import VeriTile.Triton.DSL.CallSignature
namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
noncomputable section

def callSignature : CallSignature where
  pointers := [("Q",.fp32), ("K",.fp32), ("B_Loc",.i32), ("B_Start_Loc",.i32), ("B_Seqlen",.i32), ("Att_Out",.fp32)]
  scalars := [("sm_scale",.fp32), ("max_input_len",.integer), ("stride_b_loc_b",.integer), ("stride_b_loc_s",.integer), ("stride_qbs",.integer), ("stride_qh",.integer), ("stride_qd",.integer), ("stride_kbs",.integer), ("stride_kh",.integer), ("stride_kd",.integer), ("att_stride_h",.integer), ("att_stride_bs",.integer), ("kv_group_num",.integer)]
  constexpr := ["BLOCK_DMODEL","BLOCK_N"]
  naturalIntegerPointers := ["B_Loc","B_Start_Loc","B_Seqlen"]
  typedFP32Storage := Bool.true

def source (p : Parameters) : ComputeKernel :=
  let Q := p.q
  let K := p.k
  let B_Loc := p.locations
  let B_Start_Loc := p.starts
  let B_Seqlen := p.lengths
  let Att_Out := p.output
  let BLOCK_DMODEL := p.dimensions
  let BLOCK_N := p.tokens
  with_triton_call callSignature call_args [sm_scale := p.scale, max_input_len := p.maxLength, stride_b_loc_b := p.locationBatchStride, stride_b_loc_s := p.locationSequenceStride, stride_qbs := p.queryBatchStride, stride_qh := p.queryHeadStride, stride_qd := p.queryDimStride, stride_kbs := p.keyBatchStride, stride_kh := p.keyHeadStride, stride_kd := p.keyDimStride, att_stride_h := p.outputHeadStride, att_stride_bs := p.outputSequenceStride, kv_group_num := p.groups] in triton {
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

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)

    block_stard_index = start_n * BLOCK_N
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
  }

def reuseSource (p : Parameters) : ComputeKernel :=
  let Q := p.q
  let K := p.k
  let B_Loc := p.locations
  let B_Start_Loc := p.starts
  let B_Seqlen := p.lengths
  let Att_Out := p.output
  let BLOCK_DMODEL := p.dimensions
  let BLOCK_N := p.tokens
  with_triton_call callSignature call_args [sm_scale := p.scale, max_input_len := p.maxLength, stride_b_loc_b := p.locationBatchStride, stride_b_loc_s := p.locationSequenceStride, stride_qbs := p.queryBatchStride, stride_qh := p.queryHeadStride, stride_qd := p.queryDimStride, stride_kbs := p.keyBatchStride, stride_kh := p.keyHeadStride, stride_kd := p.keyDimStride, att_stride_h := p.outputHeadStride, att_stride_bs := p.outputSequenceStride, kv_group_num := p.groups] in triton {
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
  }

end
end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked

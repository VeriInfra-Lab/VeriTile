import bench.tritonbench_g.token_attn_mistral.Proofs.Sources.Parameters
import VeriTile.Triton.DSL.IntegerMinMaxLiterals

/-! Complete current original and existing affine-candidate Python bodies.
Actual signed metadata and distinct probability/gather masks are retained. -/
namespace VeriTile.Bench.Repairs.TokenAttnMistral
open VeriTile.Triton

def callSignature (dtype : FloatDType) : CallSignature where
  pointers := [("Req_to_tokens",.i32), ("B_req_idx",.i32), ("B_Start_Loc",.i32), ("B_Seqlen",.i32), ("B_Att_Start_Loc",.i32), ("B_Att_Seqlen",.i32)]
  floatingPointers := [("Prob",dtype), ("V",dtype), ("Out",dtype)]
  scalars := [("stride_b_req_idx",.integer), ("stride_b_seq_len",.integer), ("stride_b_att_start",.integer), ("stride_b_att_len",.integer), ("stride_req_to_tokens_b",.integer), ("stride_req_to_tokens_s",.integer), ("stride_ph",.integer), ("stride_pbs",.integer), ("stride_vbs",.integer), ("stride_vh",.integer), ("stride_vd",.integer), ("stride_obs",.integer), ("stride_oh",.integer), ("stride_od",.integer), ("kv_group_num",.integer), ("sliding_window",.integer)]
  constexpr := ["BLOCK_DMODEL", "BLOCK_N"]
  readPorts := ["B_Seqlen", "B_Att_Start_Loc", "B_req_idx", "B_Att_Seqlen", "Prob", "Req_to_tokens", "V"]
  writePorts := ["Out"]

def sourceKernel (dtype : FloatDType) (Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen : RegionName)
    (stride_b_req_idx stride_b_seq_len stride_b_att_start stride_b_att_len stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [stride_b_req_idx := stride_b_req_idx, stride_b_seq_len := stride_b_seq_len, stride_b_att_start := stride_b_att_start, stride_b_att_len := stride_b_att_len, stride_req_to_tokens_b := stride_req_to_tokens_b, stride_req_to_tokens_s := stride_req_to_tokens_s, stride_ph := stride_ph, stride_pbs := stride_pbs, stride_vbs := stride_vbs, stride_vh := stride_vh, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num, sliding_window := sliding_window] in triton {
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
    p_offs = cur_head * stride_ph + (cur_batch_in_all_start_index + offs_n) * stride_pbs
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
  }

def kernel (p : Parameters) : ComputeKernel :=
  sourceKernel .fp32 p.prob p.value p.output p.tokens p.request p.unusedStart p.length p.start p.attentionLength
    p.requestStride p.sequenceStride p.attentionStartStride p.attentionLengthStride p.requestBatchStride p.requestTokenStride p.probHeadStride p.probTokenStride p.valueBatchStride p.valueHeadStride p.valueDimStride p.outputBatchStride p.outputHeadStride p.outputDimStride p.kvGroup p.window p.width p.block

end VeriTile.Bench.Repairs.TokenAttnMistral

namespace VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked
open VeriTile.Triton TokenAttnMistral

abbrev kernel := TokenAttnMistral.kernel

def affineSourcePastedKernel (dtype : FloatDType) (Prob V Out Req_to_tokens B_req_idx B_Start_Loc B_Seqlen B_Att_Start_Loc B_Att_Seqlen : RegionName)
    (stride_b_req_idx stride_b_seq_len stride_b_att_start stride_b_att_len stride_req_to_tokens_b stride_req_to_tokens_s stride_ph stride_pbs stride_vbs stride_vh stride_vd stride_obs stride_oh stride_od kv_group_num sliding_window BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [stride_b_req_idx := stride_b_req_idx, stride_b_seq_len := stride_b_seq_len, stride_b_att_start := stride_b_att_start, stride_b_att_len := stride_b_att_len, stride_req_to_tokens_b := stride_req_to_tokens_b, stride_req_to_tokens_s := stride_req_to_tokens_s, stride_ph := stride_ph, stride_pbs := stride_pbs, stride_vbs := stride_vbs, stride_vh := stride_vh, stride_vd := stride_vd, stride_obs := stride_obs, stride_oh := stride_oh, stride_od := stride_od, kv_group_num := kv_group_num, sliding_window := sliding_window] in triton {
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
  }

def affineKernel (p : Parameters) : ComputeKernel :=
  affineSourcePastedKernel .fp32 p.prob p.value p.output p.tokens p.request p.unusedStart p.length p.start p.attentionLength
    p.requestStride p.sequenceStride p.attentionStartStride p.attentionLengthStride p.requestBatchStride p.requestTokenStride p.probHeadStride p.probTokenStride p.valueBatchStride p.valueHeadStride p.valueDimStride p.outputBatchStride p.outputHeadStride p.outputDimStride p.kvGroup p.window p.width p.block

end VeriTile.Bench.Repairs.TokenAttnMistralGridYChecked

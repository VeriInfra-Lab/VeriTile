import bench.tritonbench_g.token_attn_llama2.Proofs.Sources.Parameters
import VeriTile.Triton.DSL.IntegerWhereLiterals
import VeriTile.Triton.Float.Structural

/-! Complete current original and existing block-origin-reuse Python bodies.
All caller metadata remains outside the literal quotations. -/
namespace VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
open VeriTile.Triton

def callSignature (dtype : FloatDType) : CallSignature where
  pointers := [("Q",match dtype with | .real => .fp64 | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5), ("K",match dtype with | .real => .fp64 | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5), ("B_Loc",.i32), ("B_Start_Loc",.i32), ("B_Seqlen",.i32), ("Att_Out",match dtype with | .real => .fp64 | .fp32 => .fp32 | .fp16 => .fp16 | .bf16 => .bf16 | .f8e4 => .f8e4 | .f8e5 => .f8e5)]
  floatingPointers := []
  scalars := [("sm_scale",.fp32), ("max_input_len",.integer), ("stride_b_loc_b",.integer), ("stride_b_loc_s",.integer), ("stride_qbs",.integer), ("stride_qh",.integer), ("stride_qd",.integer), ("stride_kbs",.integer), ("stride_kh",.integer), ("stride_kd",.integer), ("att_stride_h",.integer), ("att_stride_bs",.integer), ("kv_group_num",.integer)]
  constexpr := ["BLOCK_DMODEL", "BLOCK_N"]
  readPorts := ["B_Seqlen", "B_Start_Loc", "Q", "B_Loc", "K"]
  writePorts := ["Att_Out"]

def typedKernel (dtype : FloatDType) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [sm_scale := sm_scale, max_input_len := max_input_len, stride_b_loc_b := stride_b_loc_b, stride_b_loc_s := stride_b_loc_s, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kbs := stride_kbs, stride_kh := stride_kh, stride_kd := stride_kd, att_stride_h := att_stride_h, att_stride_bs := att_stride_bs, kv_group_num := kv_group_num] in triton {
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

def source (p : Parameters) : ComputeKernel :=
  typedKernel .fp32 p.q p.k p.scale p.locations p.starts p.lengths p.output p.maxLength
    p.locationBatchStride p.locationSequenceStride p.queryBatchStride p.queryHeadStride
    p.queryDimStride p.keyBatchStride p.keyHeadStride p.keyDimStride p.outputHeadStride
    p.outputSequenceStride p.groups p.dimensions p.tokens

def lengthWord {α : Type} [Inhabited α] (p : Parameters) (s : FP.Structural.State α) : Int :=
  (s.mem (Region.cast p.lengths) (s.pids 0)).read .int

def length {α : Type} [Inhabited α] (p : Parameters) (s : FP.Structural.State α) : Nat :=
  (lengthWord p s).toNat

def blockActive {α : Type} [Inhabited α] (p : Parameters) (s : FP.Structural.State α) : Prop :=
  (s.pids 2*p.tokens : Nat)<length p s
instance {α : Type} [Inhabited α] (p : Parameters) (s : FP.Structural.State α) :
    Decidable (blockActive p s) := inferInstanceAs (Decidable (_ < _))

end VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue

namespace VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked
open VeriTile.Triton TokenAttnLlama2.FPValue

abbrev source := TokenAttnLlama2.FPValue.source
abbrev typed := TokenAttnLlama2.FPValue.typedKernel
theorem source_unchanged : source=TokenAttnLlama2.FPValue.source := rfl

def reuseTypedKernel (dtype : FloatDType) (Q K : RegionName) (sm_scale : ℝ)
    (B_Loc B_Start_Loc B_Seqlen Att_Out : RegionName)
    (max_input_len stride_b_loc_b stride_b_loc_s stride_qbs stride_qh stride_qd stride_kbs stride_kh stride_kd att_stride_h att_stride_bs kv_group_num BLOCK_DMODEL BLOCK_N : Nat) : ComputeKernel :=
  with_triton_call (callSignature dtype) call_args [sm_scale := sm_scale, max_input_len := max_input_len, stride_b_loc_b := stride_b_loc_b, stride_b_loc_s := stride_b_loc_s, stride_qbs := stride_qbs, stride_qh := stride_qh, stride_qd := stride_qd, stride_kbs := stride_kbs, stride_kh := stride_kh, stride_kd := stride_kd, att_stride_h := att_stride_h, att_stride_bs := att_stride_bs, kv_group_num := kv_group_num] in triton {
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

def reuseSource (p : Parameters) : ComputeKernel :=
  reuseTypedKernel .fp32 p.q p.k p.scale p.locations p.starts p.lengths p.output p.maxLength
    p.locationBatchStride p.locationSequenceStride p.queryBatchStride p.queryHeadStride
    p.queryDimStride p.keyBatchStride p.keyHeadStride p.keyDimStride p.outputHeadStride
    p.outputSequenceStride p.groups p.dimensions p.tokens

end VeriTile.Bench.Repairs.TokenAttnLlama2GridAxesChecked

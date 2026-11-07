import VeriTile.Triton.DSL.CallSignature

/-! The complete original Python bodies. Pointer dtypes and invocation values
are supplied outside the quotations; integer loads remain signed int32 and all
source int64 widening operations are retained. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty
open VeriTile.Triton

def callSignature : CallSignature where
  pointers := [("Logits",.fp32),("presence_penalty",.fp32),
    ("freqency_penalty",.fp32),("repetition_penalty",.fp32),
    ("p_token_ids",.i32),("p_token_counts",.i32),("p_cumsum_seq_len",.i32)]
  scalars := [("stride_logit_b",.integer),("stride_logit_s",.integer)]
  constexpr := ["BLOCK_P"]
  readPorts := ["freqency_penalty","presence_penalty","repetition_penalty",
    "p_cumsum_seq_len","p_token_ids","p_token_counts","Logits"]
  writePorts := ["Logits"]

def sourceKernel
    (Logits presence_penalty freqency_penalty repetition_penalty : RegionName)
    (p_token_ids p_token_counts p_cumsum_seq_len : RegionName)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : ComputeKernel :=
  with_triton_call callSignature
    call_args [stride_logit_b := stride_logit_b, stride_logit_s := stride_logit_s] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)
    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch).to(tl.int64)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1).to(tl.int64)
    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P).to(tl.int64)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
                        mask=cur_batch_id_offset < cur_batch_end_index, other=0).to(tl.int64)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
                              mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    row_start_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64)
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits / cur_repetition, cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64) + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)
  }

def reciprocalKernel
    (Logits presence_penalty freqency_penalty repetition_penalty : RegionName)
    (p_token_ids p_token_counts p_cumsum_seq_len : RegionName)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) : ComputeKernel :=
  with_triton_call callSignature
    call_args [stride_logit_b := stride_logit_b, stride_logit_s := stride_logit_s] in triton {
    cur_batch = tl.program_id(0).to(tl.int64)
    cur_freqency = tl.load(freqency_penalty + cur_batch)
    cur_presence = tl.load(presence_penalty + cur_batch)
    cur_repetition = tl.load(repetition_penalty + cur_batch)
    cur_batch_start_index = tl.load(p_cumsum_seq_len + cur_batch).to(tl.int64)
    cur_batch_end_index = tl.load(p_cumsum_seq_len + cur_batch + 1).to(tl.int64)
    cur_batch_id_offset = cur_batch_start_index + tl.arange(0, BLOCK_P).to(tl.int64)
    batch_ids = tl.load(p_token_ids + cur_batch_id_offset,
                        mask=cur_batch_id_offset < cur_batch_end_index, other=0).to(tl.int64)
    batch_ids_count = tl.load(p_token_counts + cur_batch_id_offset,
                              mask=cur_batch_id_offset < cur_batch_end_index, other=0)
    row_start_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64)
    cur_offset = row_start_ptr + batch_ids
    cur_logits = tl.load(cur_offset, mask=cur_batch_id_offset < cur_batch_end_index, other=0.0)
    rep_logits = tl.where(cur_logits > 0, cur_logits * (1 / cur_repetition), cur_logits * cur_repetition)
    freq_logits = rep_logits - batch_ids_count * cur_freqency
    pre_logits = freq_logits - cur_presence
    output_ptr = Logits + cur_batch * tl.cast(stride_logit_b, tl.int64) + batch_ids
    tl.store(output_ptr, pre_logits, mask=cur_batch_id_offset < cur_batch_end_index)
  }

/-- Independent mathematical formula. -/
noncomputable def penalty (x : ℝ) (count : Int) (presence frequency repetition : ℝ) : ℝ :=
  (if 0 < x then x/repetition else x*repetition) - count*frequency - presence

end VeriTile.Bench.Repairs.ApplyPenalty

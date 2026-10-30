import bench.tritonbench_g.apply_penalty.IO
import bench.tritonbench_g.apply_penalty.Proofs.RealIO

namespace VeriTile.Bench.Repairs.ApplyPenalty.LogicalReader
open VeriTile Triton PublicIO
open VeriTile.Bench.TritonBenchG.ApplyPenalty
noncomputable section

theorem statement_iff (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) (f : LogicalIO.Inputs BLOCK_P → Fin BLOCK_P → ℝ) :
    (LogicalIO.io Logits presence_penalty freqency_penalty repetition_penalty
      p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P).Implements f ↔
      (applyPenaltyIO Logits presence_penalty freqency_penalty repetition_penalty
      p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P).Implements
        (fun p q presence frequency repetition start stop ids counts logits =>
          f ⟨p,q,presence,frequency,repetition,start,stop,ids,counts,logits⟩) := Iff.rfl

theorem correct (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    (LogicalIO.io Logits presence_penalty freqency_penalty repetition_penalty
      p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P).Implements (fun xs j =>
      (if 0 < xs.logits j then xs.logits j / xs.repetition else xs.logits j * xs.repetition) -
        (xs.counts j : ℝ) * xs.frequency - xs.presence) :=
  RealIOProof.apply_penalty_correctness Logits presence_penalty freqency_penalty repetition_penalty
      p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b stride_logit_s BLOCK_P

#axiomsClean statement_iff
#axiomsClean correct
end
end VeriTile.Bench.Repairs.ApplyPenalty.LogicalReader

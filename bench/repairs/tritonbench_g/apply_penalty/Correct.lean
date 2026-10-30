import bench.repairs.tritonbench_g.apply_penalty.Proofs.LogicalReader
import bench.repairs.tritonbench_g.apply_penalty.IO
import bench.repairs.tritonbench_g.apply_penalty.Proofs.RealIO

namespace VeriTile.Bench.Repairs.ApplyPenalty.Correct
open VeriTile.Triton PublicIO
open VeriTile.Bench.TritonBenchG.ApplyPenalty
open scoped VeriTile.Triton.LogicalIO

/-- In-place penalty update on the public metadata-dependent gather/scatter
windows. The shared judgment includes actual source safety, successful execution,
all active outputs and the exterior memory frame. -/
specification apply_penalty_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (LogicalIO.io Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun xs j =>
        (if 0 < xs.logits j then xs.logits j / xs.repetition else xs.logits j * xs.repetition) -
          (xs.counts j : ℝ) * xs.frequency - xs.presence) :=
  LogicalReader.correct Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P

#axiomsClean apply_penalty_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ApplyPenalty.Correct

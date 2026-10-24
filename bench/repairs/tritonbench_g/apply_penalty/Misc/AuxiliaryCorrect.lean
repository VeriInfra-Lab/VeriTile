import bench.repairs.tritonbench_g.apply_penalty.IO
import bench.repairs.tritonbench_g.apply_penalty.Proofs.RealIO

/-! Auxiliary Real correctness of the reciprocal candidate. The declaration
retains its historical namespace and complete contract. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.Correct
open VeriTile.Triton PublicIO
open VeriTile.Bench.TritonBenchG.ApplyPenalty
open scoped VeriTile.Triton.MetaScatterMasked2DKernelIO₁

/-- The existing reciprocal candidate implements the same independent Real
formula under exactly the same masked IO and active-write injectivity contract. -/
theorem apply_penalty_reciprocal_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (reciprocalIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ presence frequency repetition _ _ _ counts logits j =>
        TiledOptimizer.lionPenalty (logits j) (counts j : ℝ) repetition frequency presence) :=
  RealIOProof.apply_penalty_reciprocal_correctness Logits presence_penalty freqency_penalty
    repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
    stride_logit_b stride_logit_s BLOCK_P

#stmtSurfaceSubset apply_penalty_reciprocal_correctness ⊆
  [Region,TileDType.real,TileDType.nat,Spec.Real,reciprocalIO,
   MetaScatterMasked2DKernelIO₁.Implements,MetaScatterMasked2DKernelIO₁.B,TiledOptimizer.lionPenalty]
#axiomsClean apply_penalty_reciprocal_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ApplyPenalty.Correct

import bench.repairs.tritonbench_g.apply_penalty.IO
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Real
import bench.repairs.tritonbench_g.apply_penalty.Proofs.Projection
import bench.tritonbench_g.apply_penalty.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real correctness of both sources in Kernels.lean. The independent formula is
  (if logit > 0 then logit / repetition else logit * repetition)
    - count * frequency - presence,
written as TiledOptimizer.lionPenalty. The inputs are the initially loaded
logits and counts, together with the three scalar penalties for this program.
This is an in-place gather/scatter contract for symbolic block size and strides;
it requires active-write injectivity for the per-lane values and includes
termination, active-address bounds and the memory frame. Host launch composition and IEEE arithmetic are outside its scope.
As in the source, stride_logit_s is unused: each row is contiguous. -/
namespace VeriTile.Bench.Repairs.ApplyPenalty.RealIOProof
open VeriTile.Triton PublicIO
open VeriTile.Bench.TritonBenchG.ApplyPenalty
open scoped VeriTile.Triton.MetaScatterMasked2DKernelIO₁

/-- The masked contract requires distinct scatter addresses on active lanes.
Inactive padded ids may repeat. Bounds apply to the five scalar-slot loads and
the actual active gather/scatter addresses. All other memory is preserved. -/
theorem apply_penalty_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (applyPenaltyIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ presence frequency repetition _ _ _ counts logits j =>
        TiledOptimizer.lionPenalty (logits j) (counts j : ℝ) repetition frequency presence) := by
  refine MetaScatterMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact Projection.flattenOk Bool.false Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
  · intro bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB _
    exact Projection.safe_pinned Bool.false Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB
  · intro s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx
    exact Projection.region_run Bool.false Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx

/-- The masked contract requires distinct scatter addresses on active lanes.
Inactive padded ids may repeat. Bounds apply to the five scalar-slot loads and
the actual active gather/scatter addresses. All other memory is preserved. -/
theorem apply_penalty_reciprocal_correctness
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b stride_logit_s BLOCK_P : Nat) :
    Spec.Real (reciprocalIO Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      ⊨ fun _ _ presence frequency repetition _ _ _ counts logits j =>
        TiledOptimizer.lionPenalty (logits j) (counts j : ℝ) repetition frequency presence) := by
  refine MetaScatterMasked2DKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact Projection.flattenOk Bool.true Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
  · intro bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB _
    exact Projection.safe_pinned Bool.true Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      bounds s m₁ m₂ ids hm₁ hm₂ hids hp hf hr hstart hend hidsB hcountsB hlogitsB
  · intro s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx
    exact Projection.region_run Bool.true Logits presence_penalty freqency_penalty
      repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len
      stride_logit_b stride_logit_s BLOCK_P
      s presence frequency repetition start stop ids counts logits hp hf hr hs he hi hc hx


#axiomsClean apply_penalty_correctness
#axiomsClean apply_penalty_reciprocal_correctness
#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Repairs.ApplyPenalty.RealIOProof

import bench.tritonbench_g.apply_penalty.Kernels
import VeriTile.Triton

/-! The Real proof uses the algorithm projection of the repair source.
No independent executable implementation is maintained here. -/
namespace VeriTile.Bench.TritonBenchG.ApplyPenalty
open VeriTile.Triton
noncomputable section

def apply_penalty
    (Logits presence_penalty freqency_penalty repetition_penalty : Region .real)
    (p_token_ids p_token_counts p_cumsum_seq_len : Region .nat)
    (stride_logit_b _stride_logit_s BLOCK_P : Nat) :
    ComputeKernel :=
  let k := (VeriTile.Bench.Repairs.ApplyPenalty.kernel Logits presence_penalty freqency_penalty repetition_penalty p_token_ids p_token_counts p_cumsum_seq_len stride_logit_b _stride_logit_s BLOCK_P).toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end
end VeriTile.Bench.TritonBenchG.ApplyPenalty

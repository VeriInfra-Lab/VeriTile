import VeriTile.Meta.StatementAudit
import VeriTile.Triton.DSL
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Source
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealAccess
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealKeys
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealLoopFormula
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealProbabilities
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealRegisterSafety
import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealTiles
import bench.tritonbench_g.attention_score.Proofs.Mathematical_RealFrame
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealFormula
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealIteration
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoads
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealMemory
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealLoop
open VeriTile.Triton RealLoads RealMemory RealIteration
open AttentionFwdTriton3.RealScores (Frame)
open AttentionFwdTriton3.RealLoopFormula (Carry closed closed_zero closed_succ)
noncomputable section
set_option maxHeartbeats 1000000

def cursor (lower BN i : Nat) : Nat := lower+i*BN

theorem cursor_succ (lower BN i : Nat) : cursor lower BN (i+1) = cursor lower BN i+BN := by
  simp [cursor, Nat.add_mul, Nat.add_assoc]

theorem cursor_lt (lower upper BN i : Nat) (hN : 0 < BN) :
    cursor lower BN i < upper ↔ i < RealFormula.tripCount lower upper BN := by
  have hc : RealFormula.tripCount lower upper BN =
      AttentionFwdTriton3.RealKeys.blockCount lower upper BN := by
    by_cases h : lower < upper
    · simp only [RealFormula.tripCount, AttentionFwdTriton3.RealKeys.blockCount, if_pos h]
      rw [show upper-lower+BN-1 = (upper-lower-1)+BN from by omega, Nat.add_div_right _ hN]
    · simp [RealFormula.tripCount, AttentionFwdTriton3.RealKeys.blockCount, h,
        Nat.sub_eq_zero_of_le (show upper ≤ lower by omega), Nat.div_eq_of_lt (show BN-1 < BN by omega)]
  simpa only [cursor, hc] using (AttentionFwdTriton3.RealKeys.iteration_lt_count lower upper BN i hN).symm

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealLoop
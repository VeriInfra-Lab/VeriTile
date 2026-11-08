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
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoads
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealMemory
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealIteration
open VeriTile.Triton RealLoads RealMemory
open AttentionFwdTriton3.RealTiles (finite pairs dot_finite)
open AttentionFwdTriton3.RealScores (Frame)
open AttentionFwdTriton3.RealLoopFormula (Carry update)
noncomputable section
set_option maxHeartbeats 1200000
set_option linter.unusedSimpArgs false

structure Carries (s : BlockState) {BM BD : Nat} (carry : Carry BM BD) : Prop where
  center : s.regs .real [BM] "m_i" = some (⟨fun i => carry.center i.1⟩ : Tile .real [BM])
  denominator : s.regs .real [BM] "l_i" = some (finite (fun i : TileIndex [BM] => carry.denominator i.1))
  numerator : s.regs .real [BM,BD] "acc" = some (finite carry.numerator)

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealIteration
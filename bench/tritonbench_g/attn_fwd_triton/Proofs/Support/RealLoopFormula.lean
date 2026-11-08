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
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoop
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealLoopFormula
open VeriTile.Triton RealLoads RealLoop
open AttentionFwdTriton3.RealLoopFormula (Carry rowSeed closed_row)
open AttentionFwdTriton3.RealFormula (RowState result block_updates_eq_closed)
noncomputable section

theorem result_append (seed : RowState) (a b : List (List (ℝ × ℝ))) :
    result (result seed a) b = result seed (a++b) := by
  simp only [← block_updates_eq_closed, List.foldl_append]

def initial (BM BD : Nat) : Carry BM BD := ⟨fun _ => ⊥, fun _ => 1, fun _ => 0⟩

def firstLower (startM BM : Nat) (diagonal : Bool) : Nat := if diagonal then startM*BM else 0

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealLoopFormula
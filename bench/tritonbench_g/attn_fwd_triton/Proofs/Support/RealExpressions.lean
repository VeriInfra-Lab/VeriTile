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
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealExpressions
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false

def maskExpr (BM BN : Nat) : Op .bool [BM,BN] :=
  .ge .nat (.consR (.consL .nil))
    (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m"))
    (.add .nat .scalarL (.ref .nat [] "start_n")
      (.expandDim ⟨0, by simp⟩ (.ref .nat [BN] "offs_n")))

def maskTile (BM BN startM startN : Nat) : Tile .bool [BM,BN] :=
  ⟨fun i => decide (startN+i.2.1.val ≤ startM*BM+i.1.val)⟩

theorem tail_mask_iff (N startN lane : Nat) : lane < N-startN ↔ startN+lane < N := by
  omega

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealExpressions
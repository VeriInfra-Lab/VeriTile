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
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealIteration
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoads
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealLoopFormula
import bench.tritonbench_g.attn_fwd_triton.Proofs.Support.RealSetup
/-! Retained mathematical support for attn_fwd_triton. This module
provides no alternative kernel or finite-penalty source specification. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealFirstSetup
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000

def copies (BM BN BD : Nat) : List Stmt :=
  [.assign .nat [] "first_lo" (.constNat 0),
   .assign .nat [] "first_hi" (RealSetup.lowerExpr BM),
   .assign .ptr [BD,BN] "K_ptrs_first" (.ref .ptr [BD,BN] "K_ptrs"),
   .assign .ptr [] "K_scale_ptr_first" (.ref .ptr [] "K_scale_ptr"),
   .assign .ptr [BN,BD] "V_ptrs_first" (.ref .ptr [BN,BD] "V_ptrs")]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport.RealFirstSetup
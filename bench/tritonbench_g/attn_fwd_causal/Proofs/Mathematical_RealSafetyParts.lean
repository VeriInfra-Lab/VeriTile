import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealAccessSafety
import bench.tritonbench_g.attention_fwd_triton3.Sources.Mathematical_RealRegisterSafety
import bench.tritonbench_g.attention_score.Proofs.Mathematical_RealFrame

/-! Register-only source fragments and the helper load prefix. Safety follows
the actual branch structure; arithmetic outputs need no numerical assumptions.
The frame certificate preserves the pointers needed by the later value load. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyParts
open VeriTile.Triton RealLoads RealMemory RealAccessSafety
open AttentionFwdTriton3.RealAccess AttentionFwdTriton3.RealRegisterSafety
open AttentionScore
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def middleWrites : List RegName := ["qk","mask","mask_first","m_ij","p","l_ij","alpha","l_i","acc"]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyParts

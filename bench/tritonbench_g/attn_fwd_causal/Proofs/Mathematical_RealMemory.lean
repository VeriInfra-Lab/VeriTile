import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealLoads
import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealNumeric

/-! Execution of the actual helper loads and pointer advances. The helper
selector changes only the source-local pointer names; both helpers read the
same input-memory views, with their own current tile origin. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory
open VeriTile.Triton RealLoads
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000
set_option linter.unusedSimpArgs false

def keyName (second : Bool) : RegName := if second then "K_ptrs_second" else "K_ptrs_first"
def scaleName (second : Bool) : RegName := if second then "K_scale_ptr_second" else "K_scale_ptr_first"
def valueName (second : Bool) : RegName := if second then "V_ptrs_second" else "V_ptrs_first"

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealMemory

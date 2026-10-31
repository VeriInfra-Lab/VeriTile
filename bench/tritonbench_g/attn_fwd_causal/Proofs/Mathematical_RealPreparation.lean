import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealMemory

/-! Source score preparation, through the center and probability registers.
The diagonal flag adds the finite penalty; it never removes a probability.
First-helper branch selection is handled separately from the arithmetic. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPreparation
open VeriTile.Triton RealLoads RealMemory
open AttentionFwdTriton3.RealTiles (finite)
open AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxHeartbeats 800000

def maskName (second : Bool) : RegName := if second then "mask" else "mask_first"

def writes (second : Bool) : List RegName :=
  ["start_n","k_mask","k","k_scale","qk",maskName second,"m_ij","p"]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealPreparation

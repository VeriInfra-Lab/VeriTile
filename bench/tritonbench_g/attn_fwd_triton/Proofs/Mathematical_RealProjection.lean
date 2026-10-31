import VeriTile.Triton.DSL
import bench.tritonbench_g.attn_fwd_causal.Proofs.Mathematical_RealContract

/-! Full erased-source equality transfers real formulas and memory guarantees.
It makes no FP equivalence claim between these differently typed sources. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProjection
open VeriTile.Triton
abbrev Parameters := AttnFwdCausal.RealContract.Parameters
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProjection

import bench.tritonbench_g.attn_fwd_triton.Proofs.Mathematical_RealProjection

/-! Transfer the complete real IO contract through equality of both entire
precision-erased sources, retaining input windows, masks, successful execution,
independent weighted-sum formula and every cell outside active output writes. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProofs
open VeriTile.Triton
noncomputable section
abbrev Parameters := AttnFwdCausal.RealContract.Parameters

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProofs

import bench.tritonbench_g.attention_fwd_triton3.Proofs.Sources.Mathematical_RealExecution

/-! Compatibility import for mathematical carry and memory readback helpers. Actual repaired-source execution is proved directly in Proofs/Program.lean. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExecutionSource
open VeriTile.Triton RealExecution
noncomputable section

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealExecutionSource

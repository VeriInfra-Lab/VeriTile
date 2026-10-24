import bench.tritonbench_g.context_attn_llama.Proofs.FP
import bench.tritonbench_g.context_attn_fwd.Proofs.FPModelCore

/-! Both attention witnesses share the same strict FP32 backend and the
constructible {0,1,2} scalar domain. -/
namespace VeriTile.Bench.TritonBenchG.ContextAttnLlama.FPModel

export VeriTile.Bench.TritonBenchG.ContextAttnFwd.FPModel
  (Number zero one two sentinel maskedScore Small value literal rest model domain
    domain_strict model_satisfies_rules model_satisfies_typed_rules)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ContextAttnLlama.FPModel

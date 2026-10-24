import bench.repairs.tritonbench_g.rope_backward_transform.Proofs.Public

namespace VeriTile.Bench.Repairs.RopeBackwardTransform
open VeriTile.Bench.Repairs.RopeEmbedding.Memory (API)
/-- Complete checked fixed-coefficient input adjoint, both returned views,
odd tail, physical caller frame, and executable-check-derived index ABI. -/
theorem rope_backward_correct (api : API ℝ Store Byte) (r : Request) (m : Source.Memory ℝ)
    (call : Native.Preparation api id r m) (accepted : Checked.check api id r m call=Bool.true) :
    Checked.RealOutcome api r m call := Checked.whole_real api r m call accepted
#axiomsClean rope_backward_correct
end VeriTile.Bench.Repairs.RopeBackwardTransform

import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerMath

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

private theorem mul_exp_finite (z : ℝ) (a : WithBot ℝ) :
    WithBot.realMul (some z) (WithBot.realExp a) =
      some (z * (WithBot.realExp a).unbotD 0) := by
  cases a <;> rfl

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

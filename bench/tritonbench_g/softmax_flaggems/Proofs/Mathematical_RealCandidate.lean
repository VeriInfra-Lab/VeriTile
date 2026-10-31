import bench.tritonbench_g.softmax_flaggems.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Semantics.ProgramRewrite
import VeriTile.Triton.Memory.FlattenTolerant

/-! Whole-state real-interpreter transport for the discrete candidate change.
This lemma supplies no mathematical output theorem on its own. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealCandidate
open VeriTile Triton FullSource
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

inductive PrefixKind where
  | forwardNonInner | backwardNonInner | backwardInner

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealCandidate

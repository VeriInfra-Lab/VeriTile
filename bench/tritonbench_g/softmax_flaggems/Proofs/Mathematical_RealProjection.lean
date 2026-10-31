import bench.tritonbench_g.softmax_flaggems.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.Erasure

/-! Real projections of the complete source bodies. Storage casts disappear only
in this real abstraction; these equalities do not assert floating-point dtype
interchangeability or establish source fidelity for pending fp64 profiles. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealProjection
open VeriTile Triton FullSource
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false





end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealProjection

import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerOneTileIO
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealTrace

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false





#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInnerOneTile

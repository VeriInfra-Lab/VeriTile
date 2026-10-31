import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerOneTile
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealCandidate

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem oneSum_eq (sb : BlockState) (P G : RegionName)
    (M N rbase B C : Nat) (k : Fin C) :
    oneSum sb P G M N rbase B C k =
      ∑ n ∈ Finset.range B, sample sb P M N (rbase+k.val) n *
        sample sb G M N (rbase+k.val) n := by
  exact (Finset.sum_range (n := B) (fun j ↦
    sample sb P M N (rbase+k.val) j * sample sb G M N (rbase+k.val) j)).symm

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner

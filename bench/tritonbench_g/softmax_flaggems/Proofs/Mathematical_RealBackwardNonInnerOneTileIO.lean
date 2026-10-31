import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerOneTile
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealCandidate

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
open scoped BigOperators
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem oneSum_eq (sb : BlockState) (P G : RegionName)
    (row N K kbase B C : Nat) (k : Fin C) :
    oneSum sb P G row N K kbase B C k =
      ∑ n ∈ Finset.range B, sample sb P row N K n (kbase+k.val) *
        sample sb G row N K n (kbase+k.val) := by
  exact (Finset.sum_range (n := B) (fun j ↦
    sample sb P row N K j (kbase+k.val) * sample sb G row N K j (kbase+k.val))).symm

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner

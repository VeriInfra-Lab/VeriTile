import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 2000000

/-- The complete initial-memory formula retains every source padding sample.
No zero-fill or finite-padding hypothesis is introduced. -/
noncomputable def backwardInnerValue (base : BlockState) (P G : RegionName)
    (M N B : Nat) (one : Bool) (a : MemCellAddr) : ℝ :=
  RealBackwardInner.expected base P G M N (RealBackwardInner.paddedLength one N B)
    (a.2/N) (a.2%N)

noncomputable def backwardNonInnerValue (base : BlockState) (P G : RegionName)
    (N K B : Nat) (one : Bool) (a : MemCellAddr) : ℝ :=
  RealBackwardNonInner.expected base P G (a.2/K/N) N K
    (RealBackwardNonInner.paddedLength one N B) (a.2/K%N) (a.2%K)

#specNonCircular backwardInnerValue avoiding [run,exec]
#specNonCircular backwardNonInnerValue avoiding [run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid

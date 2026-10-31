import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridCore

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 2000000

noncomputable def forwardInnerValue (base : BlockState) (X : RegionName) (N : Nat)
    (a : MemCellAddr) : ℝ :=
  RealForwardInner.expected base X (a.2/N) N (a.2%N)

noncomputable def forwardNonInnerValue (base : BlockState) (X : RegionName) (N K : Nat)
    (a : MemCellAddr) : ℝ :=
  RealForwardNonInner.expected base X (a.2/K/N) N K (a.2/K%N) (a.2%K)

#specNonCircular forwardInnerValue avoiding [run,exec]
#specNonCircular forwardNonInnerValue avoiding [run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid

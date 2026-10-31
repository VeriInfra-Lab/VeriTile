import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerWriteback

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem reduced_pass2 (sb st : BlockState) (P G Y : RegionName)
    (row N K kbase B C : Nat) (h : ReducedInv sb P G row N K kbase B C st) :
    Pass2Inv sb P G Y row N K kbase B C 0 st := by
  constructor
  · intro r o _
    exact (congrFun (congrFun h.memory r) o).symm
  · intro o
    exact congrFun (congrFun h.memory P) o
  · intro o
    exact congrFun (congrFun h.memory G) o
  · exact h.pids
  · exact h.numPids
  · exact h.undef
  · exact h.columnReg
  · simpa using h.nReg
  · simpa using h.offsetReg
  · exact h.sumReg
  · intro n k hn hkC hkK hlt
    exact (Nat.not_lt_zero _ hlt).elim

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner

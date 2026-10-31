import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerOneTileIO
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerWriteback
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealLoopSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealTrace
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealCandidate
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealFlatFrame
import VeriTile.Triton.Memory.KernelCore
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardNonInnerFormula

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton FullSource

def paddedLength (one : Bool) (N B : Nat) : Nat :=
  if one then B else RealAddress.rounds N B*B

def ExecutionDomain (one : Bool) (P G Y : RegionName) (N B : Nat) : Prop :=
  if one then N≤B else 0<N ∧ 0<B ∧ P≠Y ∧ G≠Y

def IOResult (s final : BlockState) (P G Y : RegionName) (N K B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C) s final ∧
    (∀ n k, n<N → k<C → s.pids 1*C+k<K →
      final.readMem Y (offset (s.pids 0) N K n (s.pids 1*C+k)) =
        expected s P G (s.pids 0) N K (paddedLength one N B) n (s.pids 1*C+k)) ∧
    final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef

def FlatIOResult (A : FlatAlloc) (s final : BlockState) (P G Y : RegionName)
    (N K B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin
      (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C))
      (A.flattenState s) final ∧
    ∀ n k, n<N → k<C → s.pids 1*C+k<K →
      final.readMem A.flat (A.addr Y (offset (s.pids 0) N K n (s.pids 1*C+k))) =
        expected s P G (s.pids 0) N K (paddedLength one N B) n (s.pids 1*C+k)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner

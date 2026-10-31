import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerOneTileIO
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerInitialize
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerWriteback
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealLoopSafety
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealTrace
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealProjection
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealCandidate
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealFlatFrame
import VeriTile.Triton.Memory.KernelCore
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealBackwardInnerFormula

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton FullSource

def paddedLength (one : Bool) (N B : Nat) : Nat :=
  if one then B else RealAddress.rounds N B*B

def ExecutionDomain (one : Bool) (P G Y : RegionName) (N B : Nat) : Prop :=
  if one then N≤B else 0<N ∧ 0<B ∧ P≠Y ∧ G≠Y

def IOResult (s final : BlockState) (P G Y : RegionName) (M N B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin (outputFootprint Y M N (s.pids 0*C) C) s final ∧
    (∀ r n, r<C → s.pids 0*C+r<M → n<N →
      final.readMem Y (offset N (s.pids 0*C+r) n) =
        expected s P G M N (paddedLength one N B) (s.pids 0*C+r) n) ∧
    final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef

def FlatIOResult (A : FlatAlloc) (s final : BlockState) (P G Y : RegionName)
    (M N B C : Nat) (one : Bool) : Prop :=
  BlockState.WriteWithin
      (RealFlatFrame.footprint A (outputFootprint Y M N (s.pids 0*C) C))
      (A.flattenState s) final ∧
    ∀ r n, r<C → s.pids 0*C+r<M → n<N →
      final.readMem A.flat (A.addr Y (offset N (s.pids 0*C+r) n)) =
        expected s P G M N (paddedLength one N B) (s.pids 0*C+r) n

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner

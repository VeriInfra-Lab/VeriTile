import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerOneTileFlat
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardNonInnerFlat

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton FullSource

def ExecutionDomain (one : Bool) (X Y : RegionName) (N B : Nat) : Prop :=
  0 < N ∧ 0 < B ∧ (if one then N ≤ B else X ≠ Y)

def IOResult (s final : BlockState) (Y X : RegionName) (N K C : Nat) : Prop :=
  BlockState.WriteWithin (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C) s final ∧
    (∀ n k, n<N → k<C → s.pids 1*C+k<K →
      final.readMem Y (s.pids 0 * N * K + n * K + (s.pids 1*C+k)) =
        expected s X (s.pids 0) N K n (s.pids 1*C+k)) ∧
    final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef

def FlatIOResult (A : FlatAlloc) (s final : BlockState) (Y X : RegionName)
    (N K C : Nat) : Prop :=
  BlockState.WriteWithin
      (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N K (s.pids 1*C) C))
      (A.flattenState s) final ∧
    ∀ n k, n<N → k<C → s.pids 1*C+k<K →
      final.readMem A.flat (A.addr Y (s.pids 0 * N * K + n * K + (s.pids 1*C+k))) =
        expected s X (s.pids 0) N K n (s.pids 1*C+k)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner

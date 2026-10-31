import VeriTile.Meta.StatementAudit
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerOneTileFlat
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_RealForwardInnerFlat

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton FullSource

def ExecutionDomain (one : Bool) (X Y : RegionName) (N B : Nat) : Prop :=
  0 < N ∧ 0 < B ∧ (if one then N ≤ B else X ≠ Y)

def IOResult (s final : BlockState) (Y X : RegionName) (N : Nat) : Prop :=
  BlockState.WriteWithin (outputFootprint Y (s.pids 0) N) s final ∧
    (∀ n, n<N →
      final.readMem Y (s.pids 0 * N + n) =
        expected s X (s.pids 0) N n) ∧
    final.pids=s.pids ∧ final.numPids=s.numPids ∧ final.undef=s.undef

def FlatIOResult (A : FlatAlloc) (s final : BlockState) (Y X : RegionName)
    (N : Nat) : Prop :=
  BlockState.WriteWithin
      (RealFlatFrame.footprint A (outputFootprint Y (s.pids 0) N))
      (A.flattenState s) final ∧
    ∀ n, n<N →
      final.readMem A.flat (A.addr Y (s.pids 0 * N + n)) =
        expected s X (s.pids 0) N n

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner

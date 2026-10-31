import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridForward
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridBackward
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridMetadata

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 2000000



noncomputable def forwardValue (s : BlockState) (X : RegionName) (d : Dimensions) : MemCellAddr → ℝ :=
  if 1<d.K then forwardNonInnerValue s X d.N d.K else forwardInnerValue s X d.N

noncomputable def backwardValue (s : BlockState) (P G : RegionName) (d : Dimensions)
    (c : LaunchConfig) : MemCellAddr → ℝ :=
  if 1<d.K then backwardNonInnerValue s P G d.N d.K c.B c.one
  else backwardInnerValue s P G d.M d.N c.B c.one

def FlatResult (Y : RegionName) (count : Nat) (value : MemCellAddr → ℝ)
    (s t : BlockState) : Prop :=
  BlockState.WriteWithin (fullSlice Y count) s t ∧
    (∀ o, o<count → t.readMem Y o=value (Y,o)) ∧ t.undef=s.undef

theorem coordinate_inner {M N o : Nat} (hN : 0<N) (ho : o<M*N) :
    o/N<M ∧ o%N<N ∧ o/N*N+o%N=o :=
  ⟨(Nat.div_lt_iff_lt_mul hN).2 ho,Nat.mod_lt _ hN,owner_offset _ _⟩

theorem coordinate_noninner {M N K o : Nat} (hN : 0<N) (hK : 0<K) (ho : o<M*N*K) :
    o/K/N<M ∧ o/K%N<N ∧ o%K<K ∧
      o/K/N*N*K+o/K%N*K+o%K=o := by
  have hu : o/K<M*N := (Nat.div_lt_iff_lt_mul hK).2 ho
  refine ⟨(Nat.div_lt_iff_lt_mul hN).2 hu,Nat.mod_lt _ hN,Nat.mod_lt _ hK,?_⟩
  calc
    _ = (o/K/N*N+o/K%N)*K+o%K := by ring
    _ = o := by rw [owner_offset,owner_offset]

#axiomsClean coordinate_noninner
#specNonCircular forwardValue avoiding [run,exec]
#specNonCircular backwardValue avoiding [run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid

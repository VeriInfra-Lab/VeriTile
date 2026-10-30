import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridForward
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridBackward
import bench.tritonbench_g.softmax_flaggems.Proofs.Mathematical_HostGridMetadata

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid
open VeriTile Triton FullSource
set_option maxHeartbeats 2000000

noncomputable def forwardRun (input output : Storage) (Y X : RegionName)
    (d : Dimensions) (c : LaunchConfig) (s : BlockState) : Option BlockState :=
  if 1<d.K then
    run (softmax_kernel_non_inner input output Y X d.M d.N d.K c.B c.C c.one).eraseDType.toAlgKernel
      d.M (RealAddress.rounds d.K c.C) (grid d.M (RealAddress.rounds d.K c.C)) s
  else
    run (softmax_kernel_inner input output Y X d.M d.N c.B c.one).eraseDType.toAlgKernel
      d.M 1 (grid d.M 1) s

noncomputable def backwardRun (storage : Storage) (P G Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (s : BlockState) : Option BlockState :=
  if 1<d.K then
    run (softmax_backward_kernel_non_inner storage P G Y d.M d.N d.K c.B c.C c.one).eraseDType.toAlgKernel
      d.M (RealAddress.rounds d.K c.C) (grid d.M (RealAddress.rounds d.K c.C)) s
  else
    run (softmax_backward_kernel_inner storage P G Y d.M d.N c.C c.B c.one).eraseDType.toAlgKernel
      (RealAddress.rounds d.M c.C) 1 (grid (RealAddress.rounds d.M c.C) 1) s

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

theorem forward_dispatch (input output : Storage) (base : BlockState) (Y X : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hN : 0<d.N) (hK : 0<d.K)
    (hB : 0<c.B) (hC : 0<c.C) (hOne : c.one=oneTile d.N c.B) (hXY : X≠Y) :
    ∃ final, forwardRun input output Y X d c base=some final ∧
      FlatResult Y (d.M*d.N*d.K) (forwardValue base X d) base final := by
  have hbranch : if c.one then d.N≤c.B else X≠Y := by
    rw [hOne]
    simp only [oneTile]
    split_ifs <;> simp_all
  by_cases noninner : 1<d.K
  · obtain ⟨final,he,hf,hv,hu⟩ := forward_noninner_grid input output base Y X
      d.M d.N d.K c.B c.C c.one ⟨hN,hB,hbranch⟩ hXY hC _ (fun _ ↦ Iff.rfl)
    refine ⟨final,by simpa [forwardRun,noninner] using he,hf,?_,hu⟩
    intro o ho
    obtain ⟨hr,hn,hk,hoff⟩ := coordinate_noninner hN hK ho
    have got := hv (o/d.K/d.N) (o/d.K%d.N) (o%d.K) hr hn hk
    simpa [forwardValue,noninner,forwardNonInnerValue,hoff] using got
  · have hK1 : d.K=1 := by omega
    obtain ⟨final,he,hf,hv,hu⟩ := forward_inner_grid input output base Y X
      d.M d.N c.B c.one ⟨hN,hB,hbranch⟩ hXY _ (fun _ ↦ Iff.rfl)
    refine ⟨final,by simpa [forwardRun,noninner] using he,?_,?_,hu⟩
    · simpa [hK1] using hf
    · intro o ho
      have ho' : o<d.M*d.N := by simpa [hK1] using ho
      obtain ⟨hr,hn,hoff⟩ := coordinate_inner hN ho'
      have got := hv (o/d.N) (o%d.N) hr hn
      simpa [forwardValue,noninner,forwardInnerValue,hoff] using got

theorem backward_dispatch (storage : Storage) (base : BlockState) (P G Y : RegionName)
    (d : Dimensions) (c : LaunchConfig) (hN : 0<d.N) (hK : 0<d.K)
    (hB : 0<c.B) (hC : 0<c.C) (hOne : c.one=oneTile d.N c.B) (hPY : P≠Y) (hGY : G≠Y) :
    ∃ final, backwardRun storage P G Y d c base=some final ∧
      FlatResult Y (d.M*d.N*d.K) (backwardValue base P G d c) base final := by
  have hbranch : if c.one then d.N≤c.B else 0<d.N ∧ 0<c.B ∧ P≠Y ∧ G≠Y := by
    rw [hOne]
    simp only [oneTile]
    split_ifs <;> simp_all
  by_cases noninner : 1<d.K
  · obtain ⟨final,he,hf,hv,hu⟩ := backward_noninner_grid storage base P G Y
      d.M d.N d.K c.B c.C c.one hbranch hPY hGY hC _ (fun _ ↦ Iff.rfl)
    refine ⟨final,by simpa [backwardRun,noninner] using he,hf,?_,hu⟩
    intro o ho
    obtain ⟨hr,hn,hk,hoff⟩ := coordinate_noninner hN hK ho
    have got := hv (o/d.K/d.N) (o/d.K%d.N) (o%d.K) hr hn hk
    simpa [backwardValue,noninner,backwardNonInnerValue,hoff] using got
  · have hK1 : d.K=1 := by omega
    obtain ⟨final,he,hf,hv,hu⟩ := backward_inner_grid storage base P G Y
      d.M d.N c.B c.C c.one hbranch hPY hGY hC _ (fun _ ↦ Iff.rfl)
    refine ⟨final,by simpa [backwardRun,noninner] using he,?_,?_,hu⟩
    · simpa [hK1] using hf
    · intro o ho
      have ho' : o<d.M*d.N := by simpa [hK1] using ho
      obtain ⟨hr,hn,hoff⟩ := coordinate_inner hN ho'
      have got := hv (o/d.N) (o%d.N) hr hn
      simpa [backwardValue,noninner,backwardInnerValue,hoff] using got

#axiomsClean coordinate_noninner
#axiomsClean forward_dispatch
#axiomsClean backward_dispatch
#specNonCircular forwardValue avoiding [forwardRun,backwardRun,run,exec]
#specNonCircular backwardValue avoiding [forwardRun,backwardRun,run,exec]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.HostGrid

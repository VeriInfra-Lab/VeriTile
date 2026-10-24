import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.GridSchedule
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.SafetyOneTile
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.SafetyBackwardInner
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.SafetyBackwardNoninner
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.SafetyForwardNoninner
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.KernelInterface.Proofs.SafetyForwardInnerFull

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface
open VeriTile Triton FP.Structural SourceSafety
open SoftmaxFlaggemsFiniteSeed
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
variable {α : Type} [Inhabited α]

theorem forward_source_safe (A : Algebra α) (X Y : RegionName) (d : Dimensions)
    (c : LaunchConfig) (s : State α) (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B)
    (hone : c.one=oneTile d.N c.B) (hr : s.pids 0<d.M) (bounds : RegionName → Nat)
    (hx : ∀ k,k<d.M*d.N*d.K → k<bounds X) (hy : ∀ k,k<d.M*d.N*d.K → k<bounds Y) :
    TraceSafe A bounds (FPGrid.forwardKernel X Y d c) s := by
  unfold FPGrid.forwardKernel
  split
  · cases hc : c.one
    · exact ForwardNoninner.source A s X Y d.M d.N d.K c.B c.C hB hr bounds hx hy
    · exact forward_noninner_one A s X Y d.M d.N d.K c.B c.C hB hr bounds hx hy
  · have hk : d.K=1 := by omega
    have hx' : ∀ k,k<d.M*d.N → k<bounds X := by simpa only [hk,Nat.mul_one] using hx
    have hy' : ∀ k,k<d.M*d.N → k<bounds Y := by simpa only [hk,Nat.mul_one] using hy
    cases hc : c.one
    · have hBN : c.B<d.N := by simp [hc,oneTile] at hone; omega
      exact ForwardInner.source A s X Y d.M d.N c.B hB hBN hr bounds hx' hy'
    · exact forward_inner_one A s X Y d.M d.N c.B hB hr bounds hx' hy'

theorem backward_source_safe (A : Algebra α) (P G Y : RegionName) (d : Dimensions)
    (c : LaunchConfig) (s : State α) (hK : 0<d.K) (hr : 1<d.K → s.pids 0<d.M)
    (bounds : RegionName → Nat)
    (hp : ∀ k,k<d.M*d.N*d.K → k<bounds P) (hg : ∀ k,k<d.M*d.N*d.K → k<bounds G)
    (hy : ∀ k,k<d.M*d.N*d.K → k<bounds Y) :
    TraceSafe A bounds (FPGrid.backwardKernel P G Y d c) s := by
  unfold FPGrid.backwardKernel
  split
  · cases hc : c.one
    · exact BackwardNoninner.source A s P G Y d.M d.N d.K c.B c.C (hr ‹1<d.K›) bounds hp hg hy
    · exact backward_noninner_one A s P G Y d.M d.N d.K c.B c.C (hr ‹1<d.K›) bounds hp hg hy
  · have hk : d.K=1 := by omega
    have hp' : ∀ k,k<d.M*d.N → k<bounds P := by simpa only [hk,Nat.mul_one] using hp
    have hg' : ∀ k,k<d.M*d.N → k<bounds G := by simpa only [hk,Nat.mul_one] using hg
    have hy' : ∀ k,k<d.M*d.N → k<bounds Y := by simpa only [hk,Nat.mul_one] using hy
    cases hc : c.one
    · exact BackwardInner.source A s P G Y d.M d.N c.B c.C bounds hp' hg' hy'
    · exact backward_inner_one A s P G Y d.M d.N c.B c.C bounds hp' hg' hy'

theorem forward_safe (A : Algebra α) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hone : c.one=oneTile d.N c.B)
    (bounds : RegionName → Nat)
    (hx : ∀ k,k<d.M*d.N*d.K → k<bounds X) (hy : ∀ k,k<d.M*d.N*d.K → k<bounds Y)
    (s : State α) : ScheduleSafe bounds (forwardJobs A X Y d c) s := by
  apply jobs_safe
  intro p hp before
  have hr : p.1<d.M := by
    unfold FPGrid.forwardGrid at hp
    split at hp <;> exact (mem_grid.mp hp).1
  exact forward_source_safe A X Y d c (FPGrid.setup _ _ p before) hN hK hB hone hr bounds hx hy

theorem backward_safe (A : Algebra α) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hK : 0<d.K) (bounds : RegionName → Nat)
    (hp : ∀ k,k<d.M*d.N*d.K → k<bounds P) (hg : ∀ k,k<d.M*d.N*d.K → k<bounds G)
    (hy : ∀ k,k<d.M*d.N*d.K → k<bounds Y) (s : State α) :
    ScheduleSafe bounds (backwardJobs A P G Y d c) s := by
  apply jobs_safe
  intro p member before
  have hr : 1<d.K → p.1<d.M := by
    intro h
    exact (mem_grid.mp (by simpa [FPGrid.backwardGrid,h] using member)).1
  exact backward_source_safe A P G Y d c (FPGrid.setup _ _ p before) hK hr bounds hp hg hy

#axiomsClean forward_safe
#axiomsClean backward_safe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Interface

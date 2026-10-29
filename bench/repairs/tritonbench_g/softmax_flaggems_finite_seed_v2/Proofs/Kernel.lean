import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.NativeProfile
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.FPTraceGrid

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Kernel
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open Native
noncomputable section
set_option maxHeartbeats 6000000

def forwardTarget (R : Reductions) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (memory : RegionName → Nat → Word) : State Term :=
  SoftmaxFlaggemsFiniteSeed.FPGrid.forwardGraph (algebra R) X Y d c (entry memory)
def backwardTarget (R : Reductions) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (memory : RegionName → Nat → Word) : State Term :=
  SoftmaxFlaggemsFiniteSeed.FPGrid.backwardGraph (algebra R) P G Y d c (entry memory)

/-- All four source branches retain a family of scalar reduction replicas,
selected by each actual store consumer. There is no replica-equality premise. -/
theorem forward_complete (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (X Y : RegionName) (d : Dimensions) (c : LaunchConfig) (hB : 0<c.B)
    (memory : RegionName → Nat → Word) :
    ∃ actual, SoftmaxFlaggemsFiniteSeed.FPGrid.forwardRun (algebra R) X Y d c (entry memory)=some actual ∧
      actual.mem=(forwardTarget R X Y d c memory).mem ∧
      (∀ o, observe (Scalar.instructions ops scalar) S actual Y o=
        Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,o)
          (((forwardTarget R X Y d c memory).mem Y o).read .real))) := by
  obtain ⟨actual,he,hm⟩ := SoftmaxFlaggemsFiniteSeed.FPGrid.forward_whole (algebra R) X Y d c hB (entry memory) (entry memory) rfl
  refine ⟨actual,he,hm,?_⟩
  intro o
  rw [NativeProfile.observe_exact,hm]
  rfl

theorem backward_complete (ops : NativeProfile.Operations) (scalar : ℝ → Word) (R : Reductions) (S : Schedule)
    (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig) (memory : RegionName → Nat → Word) :
    ∃ actual, SoftmaxFlaggemsFiniteSeed.FPGrid.backwardRun (algebra R) P G Y d c (entry memory)=some actual ∧
      actual.mem=(backwardTarget R P G Y d c memory).mem ∧
      (∀ o, observe (Scalar.instructions ops scalar) S actual Y o=
        Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,o)
          (((backwardTarget R P G Y d c memory).mem Y o).read .real))) := by
  obtain ⟨actual,he,hm⟩ := SoftmaxFlaggemsFiniteSeed.FPGrid.backward_whole (algebra R) P G Y d c (entry memory) (entry memory) rfl
  refine ⟨actual,he,hm,?_⟩
  intro o
  rw [NativeProfile.observe_exact,hm]
  rfl

theorem forward_frame (I : Scalar.Instructions) (R : Reductions) (S : Schedule)
    (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (memory : RegionName → Nat → Word) (actual : State Term)
    (hm : actual.mem=(forwardTarget R X Y d c memory).mem) (r : RegionName) (o : Nat)
    (outside : ¬(r=Y ∧ o<d.M*d.N*d.K)) : observe I S actual r o=memory r o := by
  unfold observe
  rw [hm,forwardTarget,SoftmaxFlaggemsFiniteSeed.FPTrace.forward_frame (algebra R) X Y d c hN hK hB hC hone (entry memory) r o outside]
  rfl

theorem backward_frame (I : Scalar.Instructions) (R : Reductions) (S : Schedule)
    (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (memory : RegionName → Nat → Word) (actual : State Term)
    (hm : actual.mem=(backwardTarget R P G Y d c memory).mem) (r : RegionName) (o : Nat)
    (outside : ¬(r=Y ∧ o<d.M*d.N*d.K)) : observe I S actual r o=memory r o := by
  unfold observe
  rw [hm,backwardTarget,SoftmaxFlaggemsFiniteSeed.FPTrace.backward_frame (algebra R) P G Y d c hN hK hB hC hone (entry memory) r o outside]
  rfl

#axiomsClean forward_complete
#axiomsClean backward_complete
#axiomsClean forward_frame
#axiomsClean backward_frame
#specNonCircular forwardTarget avoiding [SoftmaxFlaggemsFiniteSeed.FPGrid.forwardRun,FP.Structural.exec]
#specNonCircular backwardTarget avoiding [SoftmaxFlaggemsFiniteSeed.FPGrid.backwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.Kernel

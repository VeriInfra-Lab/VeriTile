import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Kernel
import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed_v2.Proofs.Replica32

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.RawFPEquiv
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open Native
noncomputable section

/-- Complete source, consumer-indexed native outputs and all exterior cells
on raw launch geometry and fixed stored words. -/
theorem kernel_forward (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (X Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (memory : RegionName → Nat → Word) :
    ∃ actual,SoftmaxFlaggemsFiniteSeed.FPGrid.forwardRun (algebra R) X Y d c (entry memory)=some actual ∧
      actual.mem=(Kernel.forwardTarget R X Y d c memory).mem ∧
      (∀ o,observe (Scalar.instructions ops scalar) S actual Y o=
        Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,o)
          (((Kernel.forwardTarget R X Y d c memory).mem Y o).read .real))) ∧
      SoftmaxFlaggemsFiniteSeed.FPTrace.Frame Y (d.M*d.N*d.K) (entry memory) actual := by
  obtain ⟨actual,run,equal,values⟩ := Kernel.forward_complete ops scalar R S X Y d c hB memory
  refine ⟨actual,run,equal,values,?_⟩
  intro r o outside
  rw [equal,Kernel.forwardTarget,
    SoftmaxFlaggemsFiniteSeed.FPTrace.forward_frame (algebra R) X Y d c hN hK hB hC hone (entry memory) r o outside]

/-- Complete source, consumer-indexed native outputs and all exterior cells
on raw launch geometry and fixed stored words. -/
theorem kernel_backward (ops : NativeProfile.Operations) (scalar : ℝ → Word)
    (R : Reductions) (S : Schedule) (P G Y : RegionName) (d : Dimensions) (c : LaunchConfig)
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C) (hone : c.one=oneTile d.N c.B)
    (memory : RegionName → Nat → Word) :
    ∃ actual,SoftmaxFlaggemsFiniteSeed.FPGrid.backwardRun (algebra R) P G Y d c (entry memory)=some actual ∧
      actual.mem=(Kernel.backwardTarget R P G Y d c memory).mem ∧
      (∀ o,observe (Scalar.instructions ops scalar) S actual Y o=
        Ksoftmax.Graph.eval ops (NativeProfile.lower scalar S (Y,o)
          (((Kernel.backwardTarget R P G Y d c memory).mem Y o).read .real))) ∧
      SoftmaxFlaggemsFiniteSeed.FPTrace.Frame Y (d.M*d.N*d.K) (entry memory) actual := by
  obtain ⟨actual,run,equal,values⟩ := Kernel.backward_complete ops scalar R S P G Y d c memory
  refine ⟨actual,run,equal,values,?_⟩
  intro r o outside
  rw [equal,Kernel.backwardTarget,
    SoftmaxFlaggemsFiniteSeed.FPTrace.backward_frame (algebra R) P G Y d c hN hK hB hC hone (entry memory) r o outside]

#axiomsClean kernel_forward
#axiomsClean kernel_backward
#specNonCircular Kernel.forwardTarget avoiding [SoftmaxFlaggemsFiniteSeed.FPGrid.forwardRun,FP.Structural.exec]
#specNonCircular Kernel.backwardTarget avoiding [SoftmaxFlaggemsFiniteSeed.FPGrid.backwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.RawFPEquiv

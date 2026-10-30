import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardBlock
import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardBlock
import bench.tritonbench_g.softmax_flaggems.Kernels

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.RawCorrect
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
open SoftmaxFlaggemsFiniteSeed
noncomputable section

/-- Complete source, independent normalized mass and every exterior cell on
raw regions. Defined retains finite values and negative infinity in Real scope. -/
theorem kernel_forward (X Y : RegionName) (separate : X≠Y)
    (d : Dimensions) (c : LaunchConfig) (s : State (WithBot ℝ))
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C)
    (hone : c.one=oneTile d.N c.B) (defined : RealForward.Defined (FPReference.read s X) d) :
    Spec.Real (∃ t,FPGrid.forwardRun Ideal.algebra X Y d c s=some t ∧
      ∀ r o,t.mem r o=if r=Y ∧ o<d.M*d.N*d.K then
        .mk .real (Target.flatForward (FPReference.read s X) d.N d.K o : WithBot ℝ)
        else s.mem r o) := by
  obtain ⟨t,run,memory⟩ := FPGrid.forward_whole Ideal.algebra X Y d c hB s s rfl
  have target := (RealForward.whole_graph X Y (FPReference.read s X) separate
    d c defined hN hK hB hC hone s (fun _ => rfl)).2
  refine ⟨t,run,?_⟩
  intro r o
  rw [memory,target]
  rfl

/-- Complete backward source, saved-output VJP and the full regional frame. -/
theorem kernel_backward (P G Y : RegionName) (hPY : P≠Y) (hGY : G≠Y)
    (d : Dimensions) (c : LaunchConfig) (p g : Nat → ℝ) (s : State (WithBot ℝ))
    (hN : 0<d.N) (hK : 0<d.K) (hB : 0<c.B) (hC : 0<c.C)
    (hone : c.one=oneTile d.N c.B) (input : RealBackward.Input P G p g (d.M*d.N*d.K) s) :
    Spec.Real (∃ t,FPGrid.backwardRun Ideal.algebra P G Y d c s=some t ∧
      ∀ r o,t.mem r o=if r=Y ∧ o<d.M*d.N*d.K then
        .mk .real (Target.flatBackward p g d.N d.K o : WithBot ℝ) else s.mem r o) := by
  obtain ⟨t,run,memory⟩ := FPGrid.backward_whole Ideal.algebra P G Y d c s s rfl
  have target := (RealBackward.whole_graph P G Y p g hPY hGY d c
    hN hK hB hC hone s input).2
  refine ⟨t,run,?_⟩
  intro r o
  rw [memory,target]
  rfl

#axiomsClean kernel_forward
#axiomsClean kernel_backward
#specNonCircular Target.flatForward avoiding [FPGrid.forwardRun,FP.Structural.exec]
#specNonCircular Target.flatBackward avoiding [FPGrid.backwardRun,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeedV2.RawCorrect

import bench.repairs.tritonbench_g.softmax_flaggems_finite_seed.Proofs.RealTensor

namespace VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Correct
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.SoftmaxFlaggems HostGrid
noncomputable section
set_option maxHeartbeats 6000000

/-- Unchanged wrapper admission, actual launch geometry, complete readable
forward kernel, independent normalized mass and the allocated Tensor/frame.
Mathematical scope permits finite values and -infinity with one finite value
per column; it is not a runtime guard. Native endpoints admit every raw word. -/
specification checked_forward (x y : VeriTile.Torch.TensorMetadata) (dim : Int)
    (dtype : Option VeriTile.Torch.DType) (admitted : ForwardContract x dim dtype)
    (allocated : FPEquiv.Allocated x y) (ax : Fin x.shape.length)
    (hnorm : normalizeDim x.shape dim=some ax) (sms : Nat) (waves : WavesAboveOne)
    (words : Nat → BitVec 32) (initial : State (WithBot ℝ))
    (mathematical : RealTensor.MathematicalInput x ax "x" words initial) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ normalizeDim x.shape dim=some ax ∧
      ∃ result, RealTensor.forward x y ax "x" "out" (forwardConfig waves (dimensions x.shape ax) sms) initial=some result ∧
        result.tensor=y ∧ result.tensor.shape=x.shape ∧ result.tensor.device=x.device ∧ result.tensor.dtype=.floating .fp32 ∧
        RealTensor.Output "out" x.numel
          (Target.flatForward (FPReference.read initial "x") (dimensions x.shape ax).N (dimensions x.shape ax).K)
          initial result.state ∧
        RealBackward.Read "out"
          (Target.flatForward (FPReference.read initial "x") (dimensions x.shape ax).N (dimensions x.shape ax).K)
          x.numel result.state ∧
        FPTrace.Frame "out" x.numel initial result.state) := by
  obtain ⟨result,he,hm,values⟩ := RealTensor.forward_complete x y ax "x" "out" (by decide)
    (forwardConfig waves (dimensions x.shape ax) sms) initial
    (forwardConfig_positive waves _ sms).1 (forwardConfig_positive waves _ sms).2
    (forwardConfig_one waves _ sms) mathematical.2
  exact ⟨(checkForward_iff x dim dtype).mpr admitted,hnorm,result,he,hm,
    by simpa only [hm] using allocated.1,by simpa only [hm] using allocated.2.1,
    by simpa only [hm] using allocated.2.2.1,values,
    RealTensor.output_values _ _ _ _ _ values,RealTensor.output_frame _ _ _ _ _ values⟩

/-- Every backward tuning configuration executes the complete source and
returns the conventional VJP of its actual finite saved-output/upstream reads.
The forward theorem supplies the saved-output premise for normalized logits. -/
specification checked_backward (x grad y : VeriTile.Torch.TensorMetadata) (dim : Int)
    (dtype : Option VeriTile.Torch.DType) (admitted : ForwardContract x dim dtype)
    (gradient : GradientContract x grad) (allocated : FPEquiv.Allocated x y)
    (ax : Fin x.shape.length) (hnorm : normalizeDim x.shape dim=some ax) (choice : Fin 6)
    (p g : Nat → ℝ) (initial : State (WithBot ℝ))
    (input : RealBackward.Input "out" "grad" p g x.numel initial) :
    Spec.Real (checkForward x dim dtype=.ok () ∧ checkGradient x grad=.ok () ∧ normalizeDim x.shape dim=some ax ∧
      ∃ result, RealTensor.backward x y ax "out" "grad" "in_grad" (backwardConfig (dimensions x.shape ax) choice) initial=some result ∧
        result.tensor=y ∧ result.tensor.shape=x.shape ∧ result.tensor.device=x.device ∧ result.tensor.dtype=.floating .fp32 ∧
        RealTensor.Output "in_grad" x.numel
          (Target.flatBackward p g (dimensions x.shape ax).N (dimensions x.shape ax).K) initial result.state ∧
        FPTrace.Frame "in_grad" x.numel initial result.state) := by
  obtain ⟨result,he,hm,values⟩ := RealTensor.backward_complete x y ax "out" "grad" "in_grad"
    (by decide) (by decide) p g (backwardConfig (dimensions x.shape ax) choice) initial
    (backwardConfig_positive _ choice).1 (backwardConfig_positive _ choice).2 (backwardConfig_one _ choice) input
  exact ⟨(checkForward_iff x dim dtype).mpr admitted,(checkGradient_iff x grad).mpr gradient,
    hnorm,result,he,hm,by simpa only [hm] using allocated.1,
    by simpa only [hm] using allocated.2.1,by simpa only [hm] using allocated.2.2.1,
    values,RealTensor.output_frame _ _ _ _ _ values⟩

/-- Conventional column-coordinate VJP when backward consumes the proven saved
forward output, separate from any source accumulation or launch schedule. -/
theorem saved_vjp (q : Nat → WithBot ℝ) (g : Nat → ℝ) (N K row col : Nat)
    (n : Fin N) (hk : col<K) :
    Target.flatBackward (Target.flatForward q N K) g N K (Target.address N K row col n.val)=
      Target.backward q g N K row col n := by
  have hK : 0<K := by omega
  have coords := Target.coordinates N K row col n.val (by have := n.isLt; omega) hK n.isLt hk
  unfold Target.flatBackward Target.backward Target.vjp
  rw [coords.1,coords.2]
  simp_rw [Target.flatForward_coordinate q N K row col _ hK hk]

#axiomsClean checked_forward
#axiomsClean checked_backward
#axiomsClean saved_vjp
#specNonCircular Target.flatForward avoiding [RealTensor.forward,FPGrid.forwardRun,FP.Structural.exec]
#specNonCircular Target.flatBackward avoiding [RealTensor.backward,FPGrid.backwardRun,FP.Structural.exec]
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SoftmaxFlaggemsFiniteSeed.Correct

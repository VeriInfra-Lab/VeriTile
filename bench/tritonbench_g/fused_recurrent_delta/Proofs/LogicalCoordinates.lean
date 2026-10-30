import bench.tritonbench_g.fused_recurrent_delta.MathematicalIO
import bench.tritonbench_g.fused_recurrent_delta.IO
import bench.tritonbench_g.fused_recurrent_delta.Proofs.RuntimeCorrect

namespace VeriTile.Bench.Repairs.FusedRecurrentDelta.LogicalProof
open VeriTile.Triton FP.Structural Interface
noncomputable section

theorem forward_iff (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (target : ForwardInputs ℝ s → ForwardOutputs ℝ s) :
    (forwardLogicalIO s f scale bh iv).Implements
      (fun xs => (forwardOutputCoordinates ℝ s) (target ((forwardInputCoordinates ℝ s).symm xs))) ↔
    (RuntimeIO.io KernelReal.algebra .forward Bool.false s f scale bh iv).Implements target := by
  have eq : (forwardLogicalIO s f scale bh iv).lift
      (fun xs => (forwardOutputCoordinates ℝ s) (target ((forwardInputCoordinates ℝ s).symm xs))) = target := by
    funext xs
    change (forwardOutputCoordinates ℝ s).symm
      ((forwardOutputCoordinates ℝ s) (target ((forwardInputCoordinates ℝ s).symm ((forwardInputCoordinates ℝ s) xs)))) = target xs
    rw [Equiv.symm_apply_apply,Equiv.symm_apply_apply]
  change (RuntimeIO.io KernelReal.algebra .forward Bool.false s f scale bh iv).Implements _ ↔ _
  rw [eq]

theorem backward_iff (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (domain : BackwardInputs ℝ s → Prop) (target : BackwardInputs ℝ s → BackwardOutputs ℝ s) :
    (backwardLogicalIO s f scale bh iv).ImplementsOn
      (fun xs => domain ((backwardInputCoordinates ℝ s).symm xs))
      (fun xs => (backwardOutputCoordinates ℝ s) (target ((backwardInputCoordinates ℝ s).symm xs))) ↔
    (RuntimeIO.io KernelReal.algebra .backward Bool.false s f scale bh iv).ImplementsOn domain target := by
  have eq : (backwardLogicalIO s f scale bh iv).lift
      (fun xs => (backwardOutputCoordinates ℝ s) (target ((backwardInputCoordinates ℝ s).symm xs))) = target := by
    funext xs
    change (backwardOutputCoordinates ℝ s).symm
      ((backwardOutputCoordinates ℝ s) (target ((backwardInputCoordinates ℝ s).symm ((backwardInputCoordinates ℝ s) xs)))) = target xs
    rw [Equiv.symm_apply_apply,Equiv.symm_apply_apply]
  have deq : (fun xs => domain ((backwardInputCoordinates ℝ s).symm ((backwardInputCoordinates ℝ s) xs))) = domain := by
    funext xs
    rw [Equiv.symm_apply_apply]
  change (RuntimeIO.io KernelReal.algebra .backward Bool.false s f scale bh iv).ImplementsOn
    (fun xs => domain ((backwardInputCoordinates ℝ s).symm ((backwardInputCoordinates ℝ s) xs))) _ ↔ _
  simp only [Equiv.symm_apply_apply,eq]

theorem forward (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat) :
    (forwardLogicalIO s f scale bh iv).Implements
      (fun xs => (forwardOutputCoordinates ℝ s) (forwardFormula s f scale ((forwardInputCoordinates ℝ s).symm xs))) :=
  (forward_iff s f scale bh iv _).mpr (RuntimeCorrect.forward s f scale bh iv)

theorem backward (s : Shape) (f : Flags) (scale : ℝ) (bh iv : Nat)
    (a : Fin s.BBV → Math.Input s.BK) (g : Nat → Fin s.BBV → ℝ) (terminal : Fin s.BBV → Math.Vec s.BK) :
    (backwardLogicalIO s f scale bh iv).ImplementsOn (savedDomain s f iv a g terminal)
      (fun xs => (backwardOutputCoordinates ℝ s) (backwardFormula s scale a g terminal ((backwardInputCoordinates ℝ s).symm xs))) :=
  (backward_iff s f scale bh iv _ _).mpr (RuntimeCorrect.backward s f scale bh iv a g terminal)
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta.LogicalProof

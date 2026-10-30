import bench.tritonbench_g.cross_entropy2.IO
import bench.tritonbench_g.cross_entropy2.Proofs.RealIO

namespace VeriTile.Bench.Repairs.CrossEntropy2.LogicalProof
open VeriTile.Triton FP.Structural
open CrossEntropy1.LogicalData
noncomputable section

theorem forward_transport (p : Parameters) (row block : Nat)
    (f : (Nat → ℝ) → Int → ℝ × ℝ × ℝ) :
    (LogicalIO.forward p row block).lift (forwardTargetZ f) = IO.forwardFormula p f := by
  funext vals o j
  fin_cases o <;> cases j <;> rfl

theorem backward_transport (p : Parameters) (row block : Nat)
    (f : (Nat → ℝ) → Int → ℝ → ℝ → Fin p.block → ℝ) :
    (LogicalIO.backward p row block).lift (backwardTarget f) = IO.backwardFormula p f := by
  funext vals o j
  fin_cases o
  rcases j with ⟨i,u⟩
  cases u
  rfl

theorem forward_iff (p : Parameters) (row block : Nat)
    (domain : ForwardInputs ℝ p.cols → Prop) (f : (Nat → ℝ) → Int → ℝ × ℝ × ℝ) :
    (LogicalIO.forward p row block).ImplementsOn domain (forwardTargetZ f) ↔
      (LogicalIO.forward p row block).core.ImplementsOn
        (fun raw => domain ((LogicalIO.forward p row block).inputs raw)) (IO.forwardFormula p f) :=
  Iff.of_eq (congrArg ((LogicalIO.forward p row block).core.ImplementsOn
    (fun raw => domain ((LogicalIO.forward p row block).inputs raw))) (forward_transport p row block f))

theorem backward_iff (p : Parameters) (row block : Nat)
    (f : (Nat → ℝ) → Int → ℝ → ℝ → Fin p.block → ℝ) :
    (LogicalIO.backward p row block).Implements (backwardTarget f) ↔
      (LogicalIO.backward p row block).core.Implements (IO.backwardFormula p f) :=
  Iff.of_eq (congrArg (LogicalIO.backward p row block).core.Implements (backward_transport p row block f))

theorem forward (p : Parameters) (row block : Nat) (hr : row<p.rows)
    (positive : 0<p.block) (start : block*p.block<p.cols) (separate : p.lse≠p.logits)
    (outputs : p.lse≠p.loss ∧ p.lse≠p.zLoss ∧ p.loss≠p.zLoss)  :
    (LogicalIO.forward p row block).Implements (forwardTargetZ (fun xs label => (KernelMath.lse p xs block,KernelMath.loss p xs label block,KernelMath.zLoss p xs label block))) :=
  (forward_iff p row block (fun _ => True) (fun xs label => (KernelMath.lse p xs block,KernelMath.loss p xs label block,KernelMath.zLoss p xs label block))).mpr (RealIO.forward p row block hr positive start separate outputs)

theorem backward (p : Parameters) (row block : Nat) (hr : row<p.rows)  :
    (LogicalIO.backward p row block).Implements (backwardTarget (fun xs label L upstream lane => KernelMath.gradient p xs label L upstream (block*p.block+lane.val))) :=
  (backward_iff p row block (fun xs label L upstream lane => KernelMath.gradient p xs label L upstream (block*p.block+lane.val))).mpr (RealIO.backward p row block hr)

end
end VeriTile.Bench.Repairs.CrossEntropy2.LogicalProof

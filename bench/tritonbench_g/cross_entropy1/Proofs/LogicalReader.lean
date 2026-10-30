import bench.tritonbench_g.cross_entropy1.IO
import bench.tritonbench_g.cross_entropy1.Proofs.RealIO

namespace VeriTile.Bench.Repairs.CrossEntropy1.LogicalProof
open VeriTile.Triton FP.Structural
open CrossEntropy1.LogicalData
noncomputable section

theorem forward_transport (p : Parameters) (row block : Nat)
    (f : (Nat → ℝ) → Int → ℝ × ℝ) :
    (LogicalIO.forward p row block).lift (forwardTarget f) = IO.forwardFormula p f := by
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
    (domain : ForwardInputs ℝ p.cols → Prop) (f : (Nat → ℝ) → Int → ℝ × ℝ) :
    (LogicalIO.forward p row block).ImplementsOn domain (forwardTarget f) ↔
      (LogicalIO.forward p row block).core.ImplementsOn
        (fun raw => domain ((LogicalIO.forward p row block).inputs raw)) (IO.forwardFormula p f) :=
  Iff.of_eq (congrArg ((LogicalIO.forward p row block).core.ImplementsOn
    (fun raw => domain ((LogicalIO.forward p row block).inputs raw))) (forward_transport p row block f))

theorem backward_iff (p : Parameters) (row block : Nat)
    (f : (Nat → ℝ) → Int → ℝ → ℝ → Fin p.block → ℝ) :
    (LogicalIO.backward p row block).Implements (backwardTarget f) ↔
      (LogicalIO.backward p row block).core.Implements (IO.backwardFormula p f) :=
  Iff.of_eq (congrArg (LogicalIO.backward p row block).core.Implements (backward_transport p row block f))

theorem forward_unsplit (p : Parameters) (row : Nat) (hr : row<p.rows)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (positive : 0<p.cols) (covered : p.cols≤p.block) (splitFlag : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols)  :
    (LogicalIO.forward p row 0).ImplementsOn (LogicalIO.unsplitDomain p) (forwardTarget (fun xs label => (Math.lse p.cols xs,Math.loss p.cols xs label p.ignored p.smoothing p.squareScale))) :=
  (forward_iff p row 0 (LogicalIO.unsplitDomain p) (fun xs label => (Math.lse p.cols xs,Math.loss p.cols xs label p.ignored p.smoothing p.squareScale))).mpr (RealIO.forward_unsplit p row hr separate outputs positive covered splitFlag noSmooth startZero total)

theorem forward_split (p : Parameters) (row block : Nat) (hr : row<p.rows)
    (separate : p.lse≠p.logits) (outputs : p.lse≠p.loss)
    (positive : 0<p.block) (start : block*p.block<p.cols) (splitFlag : p.split=Bool.true)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)  :
    (LogicalIO.forward p row block).Implements (forwardTarget (fun xs label => (Math.blockLSE p.cols p.block block xs,RealSource.partialLoss p xs label block))) :=
  (forward_iff p row block (fun _ => True) (fun xs label => (Math.blockLSE p.cols p.block block xs,RealSource.partialLoss p xs label block))).mpr (RealIO.forward_split p row block hr separate outputs positive start splitFlag noSmooth)

theorem backward (p : Parameters) (row block : Nat)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)  :
    (LogicalIO.backward p row block).Implements (backwardTarget (fun xs label L upstream lane => KernelMath.gradient p xs label L upstream (block*p.block+lane.val))) :=
  (backward_iff p row block (fun xs label L upstream lane => KernelMath.gradient p xs label L upstream (block*p.block+lane.val))).mpr (RealIO.backward p row block noSmooth)

end
end VeriTile.Bench.Repairs.CrossEntropy1.LogicalProof

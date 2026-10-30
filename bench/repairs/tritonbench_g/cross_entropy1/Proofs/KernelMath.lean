import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelContract
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.Real

/-! Independent kernel mathematics linked to actual logical tensor inputs.
The following results are deliberately named `scalar_*`: their execution is
the old scalar interpreter, not the typed DSL. They are useful bridge targets,
not current public kernel-correctness claims. -/
namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelMath
open VeriTile.Triton FP.Structural Source KernelContract
open TensorModel (Tensor)
noncomputable section
set_option maxHeartbeats 4000000

/-- Backward's LSE is a legitimate input to this JIT. It need not be proved by
a host forward/aggregation theorem to state this local kernel specification. -/
def gradient (p : Parameters) (x : Nat → ℝ) (target : Int)
    (lse upstream : ℝ) (col : Nat) : ℝ :=
  if target=p.ignored then 0 else upstream*
    ((1+2*p.squareScale*lse)*Real.exp (x col-lse)-
      (if (col : Int)=target-p.classStart then 1-p.smoothing else 0)-
      p.smoothing/p.totalClasses)

theorem row_input (p : Parameters) (initial : State (WithBot ℝ))
    (t : Tensor) (view : MatrixView t p.logits p.logitsBase p.rows p.cols p.logitsStride)
    (row : Nat) (hr : row<p.rows) (x : Nat → ℝ)
    (values : ∀ col, col<p.cols → fpValue initial t (row*p.cols+col)=↑(x col)) :
    RealSource.Input p (scalarMemory initial) row x := by
  intro col hc
  exact (logit_binding p initial t view row col hr hc).trans (values col hc)

/-- Both local forward stores and every untouched tagged cell are accounted
for, with the target obtained from the actual integer tensor. -/
theorem scalar_forward_unsplit (p : Parameters) (views : ForwardViews p)
    (initial : State (WithBot ℝ)) (row : Nat) (hr : row<p.rows)
    (x : Nat → ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (positive : 0<p.cols) (covered : p.cols≤p.block) (splitFlag : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (startZero : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : targetValue initial views.labels row=p.ignored ∨
      (0≤targetValue initial views.labels row ∧ targetValue initial views.labels row<(p.cols : Int))) :
    let final := Source.forward RealSource.algebra p (scalarMemory initial) row 0
    (final p.lse (p.lseBase+row)).read .real=↑(Math.lse p.cols x) ∧
    (final p.loss (p.lossBase+row)).read .real=
      ↑(Math.loss p.cols x (targetValue initial views.labels row) p.ignored p.smoothing p.squareScale) ∧
    (∀ region i, (region≠p.loss ∨ i≠p.lossBase+row) →
      (region≠p.lse ∨ i≠p.lseBase+row) → final region i=scalarMemory initial region i) := by
  have input := row_input p initial views.logits views.logitsView row hr x values
  have target := label_binding p initial views.labels views.labelsView row hr
  have outputs := Program.forward_outputs RealSource.algebra p (scalarMemory initial) row 0
    views.distinctOutputs views.lseLogits
  have loss := RealSource.unsplit_loss p (scalarMemory initial) row x input
    (targetValue initial views.labels row) positive covered splitFlag noSmooth startZero total valid
  have lse := RealSource.forward_lse p (scalarMemory initial) row 0 x input
    (by omega) (by simpa using positive)
  rw [Math.block_zero_lse p.cols p.block covered] at lse
  simp only [Nat.zero_mul,Nat.add_zero,target] at outputs
  refine ⟨outputs.1.trans lse,outputs.2.trans loss,?_⟩
  intro region i lossMiss lseMiss
  exact Program.forward_frame RealSource.algebra p (scalarMemory initial) row 0 region i
    (by simpa only [Nat.zero_mul,Nat.add_zero] using lossMiss)
    (by simpa only [Nat.zero_mul,Nat.add_zero] using lseMiss)

/-- Split kernels store only their local contribution; the block LSE is a
second independent output. No Torch aggregation is part of this result. -/
theorem scalar_forward_split (p : Parameters) (views : ForwardViews p)
    (initial : State (WithBot ℝ)) (row block : Nat) (hr : row<p.rows)
    (x : Nat → ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (width : 0<p.block) (start : block*p.block<p.cols) (splitFlag : p.split=Bool.true)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    let final := Source.forward RealSource.algebra p (scalarMemory initial) row block
    (final p.lse (p.lseBase+block*p.rows+row)).read .real=↑(Math.blockLSE p.cols p.block block x) ∧
    (final p.loss (p.lossBase+block*p.rows+row)).read .real=
      ↑(RealSource.partialLoss p x (targetValue initial views.labels row) block) ∧
    (∀ region i, (region≠p.loss ∨ i≠p.lossBase+block*p.rows+row) →
      (region≠p.lse ∨ i≠p.lseBase+block*p.rows+row) → final region i=scalarMemory initial region i) := by
  have input := row_input p initial views.logits views.logitsView row hr x values
  have target := label_binding p initial views.labels views.labelsView row hr
  have outputs := Program.forward_outputs RealSource.algebra p (scalarMemory initial) row block
    views.distinctOutputs views.lseLogits
  rw [target] at outputs
  exact ⟨outputs.1.trans (RealSource.forward_lse p _ row block x input width start),
    outputs.2.trans (RealSource.split_loss p _ row block x input _ splitFlag noSmooth),
    Program.forward_frame RealSource.algebra p (scalarMemory initial) row block⟩

/-- All active gradient stores and the complete frame, including an in-place
logit output. Initial scalar reads are taken through the supplied tensors. -/
theorem scalar_backward (p : Parameters) (views : BackwardViews p)
    (initial : State (WithBot ℝ)) (row block : Nat) (hr : row<p.rows)
    (x : Nat → ℝ) (lse upstream : ℝ)
    (values : ∀ col, col<p.cols → fpValue initial views.logits (row*p.cols+col)=↑(x col))
    (lseValue : fpValue initial views.lse row=↑lse)
    (upstreamValue : targetValue initial views.labels row≠p.ignored →
      fpValue initial views.upstream row=↑upstream)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    let final := Source.backward RealSource.algebra p (scalarMemory initial) row block
    (∀ lane : Fin p.block, column p block lane.val<p.cols →
      (final p.gradient (gradientAddress p row (column p block lane.val))).read .real=
        ↑(gradient p x (targetValue initial views.labels row) lse upstream (column p block lane.val))) ∧
    (∀ region offset, (∀ lane : Fin p.block, column p block lane.val<p.cols →
      region≠p.gradient ∨ offset≠gradientAddress p row (column p block lane.val)) →
      final region offset=scalarMemory initial region offset) := by
  have input := row_input p initial views.logits views.logitsView row hr x values
  have target := label_binding p initial views.labels views.labelsView row hr
  have lseRead : ((scalarMemory initial) p.lse (p.lseBase+row)).read .real=↑lse := by
    simpa only [fpValue,vector_index _ _ _ _ _ _ views.lseView hr,
      views.lseView.regionEq,Nat.mul_one,scalarMemory,Source.Cell.read] using lseValue
  refine ⟨?_,Program.backward_frame RealSource.algebra p (scalarMemory initial) row block⟩
  intro lane active
  rw [Program.backward_output RealSource.algebra p _ row block lane active]
  by_cases ignored : targetValue initial views.labels row=p.ignored
  · rw [Program.backward_ignored RealSource.algebra p _ row block (target.trans ignored) lane]
    simp only [gradient,ignored,↓reduceIte,RealSource.fp_coe]
  have upstreamRead : ((scalarMemory initial) p.upstream
      (p.upstreamBase+row*p.upstreamStride)).read .real=↑upstream := by
    simpa only [fpValue,vector_index _ _ _ _ _ _ views.upstreamView hr,
      views.upstreamView.regionEq,scalarMemory,Source.Cell.read] using upstreamValue ignored
  exact RealSource.backward_lane p _ row block x input _ lse upstream
    target lseRead upstreamRead noSmooth lane active

#axiomsClean row_input
#axiomsClean scalar_forward_unsplit
#axiomsClean scalar_forward_split
#axiomsClean scalar_backward
#specNonCircular gradient avoiding [Source.forward,Source.backward,forwardSyntax,backwardSyntax]
#specNonCircular Math.loss avoiding [Source.forward,Source.backward,forwardSyntax,backwardSyntax]
#specNonCircular RealSource.partialLoss avoiding [Source.forward,Source.backward,forwardSyntax,backwardSyntax]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelMath

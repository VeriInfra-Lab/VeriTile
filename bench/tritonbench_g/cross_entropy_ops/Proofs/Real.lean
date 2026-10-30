import bench.tritonbench_g.cross_entropy_ops.Proofs.Source
import bench.tritonbench_g.cross_entropy_ops.Proofs.Math
import bench.tritonbench_g.cross_entropy1.Proofs.Real

namespace VeriTile.Bench.Repairs.CrossEntropyOps.RealSource
open VeriTile.Triton
open CrossEntropy1.Source (Memory Cell)
noncomputable section
set_option maxHeartbeats 4000000

abbrev algebra := CrossEntropy1.RealSource.algebra
abbrev Input (p : Parameters) (memory : Memory (WithBot ℝ)) (row : Nat) (x : Nat → ℝ) :=
  CrossEntropy1.RealSource.Input p.toParameters memory row x

theorem scaled_input (p : Parameters) (memory : Memory (WithBot ℝ)) (row : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (backward : Bool) :
    CrossEntropy1.RealSource.Input p.toParameters (Source.scaledReads algebra p memory backward) row
      (Spec.scaled p.logitScale x) := by
  intro col hc
  simp only [Source.scaledReads,Cell.read,eq_self,true_or,if_true]
  change algebra.mul ((memory p.logits (CrossEntropy1.Source.logitAddress p.toParameters row col)).read .real)
    (↑p.logitScale)=↑(x col*p.logitScale)
  rw [input col hc]
  rfl

theorem forward_lse (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (positive : 0<p.block)
    (start : block*p.block<p.cols) :
    (Source.forwardValues algebra p memory row block).lse=
      ↑(CrossEntropy1.Math.blockLSE p.cols p.block block (Spec.scaled p.logitScale x)) := by
  exact CrossEntropy1.RealSource.forward_lse p.toParameters
    (Source.scaledReads algebra p memory Bool.false) row block (Spec.scaled p.logitScale x)
    (scaled_input p memory row x input Bool.false) positive start

theorem forward_split_loss (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (target : Int)
    (split : p.split=Bool.true) (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0) :
    Source.forwardLoss algebra p memory row block target (Source.forwardValues algebra p memory row block)=
      ↑(CrossEntropy1.RealSource.partialLoss p.toParameters (Spec.scaled p.logitScale x) target block) := by
  exact CrossEntropy1.RealSource.split_loss p.toParameters
    (Source.scaledReads algebra p memory Bool.false) row block (Spec.scaled p.logitScale x)
    (scaled_input p memory row x input Bool.false) target split noSmooth

theorem forward_unsplit_loss (p : Parameters) (memory : Memory (WithBot ℝ)) (row : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (target : Int)
    (positive : 0<p.cols) (covered : p.cols≤p.block) (split : p.split=Bool.false)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (start : p.classStart=0) (total : p.totalClasses=p.cols)
    (valid : target=p.ignored ∨ (0≤target ∧ target<(p.cols : Int))) :
    Source.forwardLoss algebra p memory row 0 target (Source.forwardValues algebra p memory row 0)=
      ↑(Spec.loss p.cols p.logitScale p.smoothing p.squareScale x target p.ignored) := by
  exact CrossEntropy1.RealSource.unsplit_loss p.toParameters
    (Source.scaledReads algebra p memory Bool.false) row (Spec.scaled p.logitScale x)
    (scaled_input p memory row x input Bool.false) target positive covered split noSmooth start total valid

theorem logging_zloss (p : Parameters) (target : Int) (L : ℝ) (unsplit : p.split=Bool.false) :
    Source.zLoss algebra p target (↑L)=
      ↑(if target=p.ignored then 0 else p.squareScale*L*L) := by
  by_cases h : target=p.ignored
  · simp only [Source.zLoss,if_pos h]
    rfl
  · simp only [Source.zLoss,if_neg h,unsplit,Bool.false_eq_true,if_false]
    rfl

/-- Standalone backward consumes its actual supplied saved LSE. Whole-call
composition must later identify this L with the actual forward/global result. -/
theorem backward_lane (p : Parameters) (memory : Memory (WithBot ℝ)) (row block : Nat)
    (x : Nat → ℝ) (input : Input p memory row x) (target : Int) (L upstream : ℝ)
    (labelRead : CrossEntropy1.Source.label p.toParameters memory row=target)
    (lseRead : (memory p.lse (p.lseBase+row)).read .real=↑L)
    (upstreamRead : (memory p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real=↑upstream)
    (lseNotLogits : p.lse≠p.logits) (lseNotUpstream : p.lse≠p.upstream)
    (noSmooth : p.hasSmoothing=Bool.false → p.smoothing=0)
    (lane : Fin p.block) (active : CrossEntropy1.Source.column p.toParameters block lane.val<p.cols) :
    Source.backwardValues algebra p memory row block lane=
      ↑(if target=p.ignored then 0 else (upstream*p.logitScale)*
        ((1+2*p.squareScale*L)*Real.exp (x (CrossEntropy1.Source.column p.toParameters block lane.val)*p.logitScale-L)-
          (if (CrossEntropy1.Source.column p.toParameters block lane.val : Int)=target-p.classStart
            then 1-p.smoothing else 0)-p.smoothing/p.totalClasses)) := by
  have saved : ((Source.scaledReads algebra p memory Bool.true) p.lse (p.lseBase+row)).read .real=↑L := by
    simpa [Source.scaledReads,Cell.read,lseNotLogits,lseNotUpstream] using lseRead
  have upstreamScaled : ((Source.scaledReads algebra p memory Bool.true) p.upstream
      (p.upstreamBase+row*p.upstreamStride)).read .real=↑(upstream*p.logitScale) := by
    simp only [Source.scaledReads,Cell.read,eq_self,and_self,or_true,if_true]
    change algebra.mul ((memory p.upstream (p.upstreamBase+row*p.upstreamStride)).read .real)
      (↑p.logitScale)=↑(upstream*p.logitScale)
    rw [upstreamRead]
    rfl
  exact CrossEntropy1.RealSource.backward_lane p.toParameters
    (Source.scaledReads algebra p memory Bool.true) row block (Spec.scaled p.logitScale x)
    (scaled_input p memory row x input Bool.true) target L (upstream*p.logitScale)
    labelRead saved upstreamScaled noSmooth lane active

#axiomsClean scaled_input
#axiomsClean forward_lse
#axiomsClean forward_split_loss
#axiomsClean forward_unsplit_loss
#axiomsClean logging_zloss
#axiomsClean backward_lane
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.RealSource

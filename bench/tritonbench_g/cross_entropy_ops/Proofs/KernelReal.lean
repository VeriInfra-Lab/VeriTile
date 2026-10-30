import bench.tritonbench_g.cross_entropy_ops.Proofs.KernelBackward
import bench.tritonbench_g.cross_entropy_ops.Proofs.Real
import bench.tritonbench_g.cross_entropy_ops.Proofs.KernelContract
import bench.tritonbench_g.cross_entropy1.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelReal
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

abbrev algebra := CrossEntropy1.KernelReal.algebra
attribute [local simp] CrossEntropy1.KernelReal.decimal_zero CrossEntropy1.KernelReal.decimal_one CrossEntropy1.KernelReal.decimal_two

def arguments (p : Parameters) : RuntimeArgs.Values (WithBot ℝ) := ⟨⟨↑p.smoothing,↑p.squareScale⟩,↑p.logitScale⟩

/-- The post-scale mask discards the unused padding product. This lemma is
pointwise and needs no law concerning NaN/FP arithmetic. -/
theorem logits_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) (backward : Bool) (lane : Fin p.block) :
    KernelNative.logits algebra p initial (lane,PUnit.unit)=
      CrossEntropy1.Source.loadedLogits (Source.arithmetic RealSource.algebra) p.toParameters
        (Source.scaledReads RealSource.algebra p (scalarMemory initial) backward)
        (initial.pids 0) (initial.pids 1) lane := by
  by_cases active : initial.pids 1*p.block+lane.val<p.cols
  all_goals simp [KernelNative.logits,KernelNative.rawLogits,Source.arithmetic,Source.scaledReads,
    CrossEntropy1.Source.loadedLogits,CrossEntropy1.Source.load,CrossEntropy1.Source.column,
    CrossEntropy1.Source.logitAddress,scalarMemory,CrossEntropy1.Source.Cell.read,
    algebra,CrossEntropy1.KernelReal.algebra,RealSource.algebra,CrossEntropy1.RealSource.algebra,
    RuntimeArgs.fp,abi.logitScale,arguments,active]

theorem logits_function (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) (backward : Bool) :
    KernelNative.logits algebra p initial=(fun i =>
      CrossEntropy1.Source.loadedLogits (Source.arithmetic RealSource.algebra) p.toParameters
        (Source.scaledReads RealSource.algebra p (scalarMemory initial) backward)
        (initial.pids 0) (initial.pids 1) i.1) := by
  funext ⟨lane,u⟩
  cases u
  exact logits_scalar p initial abi backward lane

theorem lse_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) :
    KernelNative.lseValue algebra p initial=
      (Source.forwardValues RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1)).lse := by
  unfold KernelNative.lseValue KernelNative.lseTile KernelNative.maxValue KernelNative.maxTile
  rw [logits_function p initial abi Bool.false]
  rfl

theorem sum_option (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) :
    (Source.forwardValues RealSource.algebra p (scalarMemory initial)
      (initial.pids 0) (initial.pids 1)).sumLogits=
      if p.hasSmoothing then some (KernelNative.sumValue algebra p initial) else none := by
  cases hs : p.hasSmoothing
  · simp [Source.forwardValues,CrossEntropy1.Source.forwardValues,hs]
  · simp only [hs,if_true]
    unfold KernelNative.sumValue KernelNative.sumTile
    rw [logits_function p initial abi Bool.false]
    simp only [Source.forwardValues,CrossEntropy1.Source.forwardValues,hs,if_true]
    simp only [algebra,CrossEntropy1.KernelReal.algebra,CrossEntropy1.KernelReal.lanes,
      Source.arithmetic,RealSource.algebra,CrossEntropy1.RealSource.algebra,
      CrossEntropy1.Source.sum,CrossEntropy1.Source.fp,CrossEntropy1.Source.column,
      CrossEntropy1.KernelReal.decimal_zero]
    rfl

@[simp] theorem target_scalar (p : Parameters) (initial : State (WithBot ℝ)) :
    KernelNative.target p initial=CrossEntropy1.Source.label p.toParameters (scalarMemory initial) (initial.pids 0) := rfl

/-- Ideal interpretation of the independent native loss graph. No execution
or output premise occurs in this arithmetic bridge. -/
theorem loss_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) :
    KernelNative.lossValue algebra p initial=
      Source.forwardLoss RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1)
        (CrossEntropy1.Source.label p.toParameters (scalarMemory initial) (initial.pids 0))
        (Source.forwardValues RealSource.algebra p (scalarMemory initial)
          (initial.pids 0) (initial.pids 1)) := by
  have lseEq := lse_scalar p initial abi
  have sumEq := sum_option p initial abi
  simp [algebra,CrossEntropy1.KernelReal.algebra,RealSource.algebra,CrossEntropy1.RealSource.algebra] at lseEq sumEq
  cases hs : p.hasSmoothing <;> cases sp : p.split
  all_goals
    simp [KernelNative.lossValue,KernelNative.coreLoss,Source.forwardLoss,CrossEntropy1.Source.forwardLoss,
      ← lseEq,sumEq,hs,sp,Source.arithmetic,
      KernelNative.owns,KernelNative.target,CrossEntropy1.Source.label,CrossEntropy1.Source.fp,CrossEntropy1.Source.nat,
      CrossEntropy1.Source.binary,CrossEntropy1.Source.load,CrossEntropy1.Source.logitAddress,KernelNative.gathered,
      Source.scaledReads,scalarMemory,CrossEntropy1.Source.Cell.read,
      algebra,CrossEntropy1.KernelReal.algebra,RealSource.algebra,CrossEntropy1.RealSource.algebra,
      RuntimeArgs.fp,RuntimeArgs.classes,abi.smoothing,abi.squareScale,abi.totalClasses,abi.logitScale,arguments]
    all_goals
      try simp only [← lseEq,sumEq,hs,if_true,Bool.false_eq_true,if_false,Option.getD_some]
      all_goals rfl

/-- Standalone backward's saved LSE is a separate input allocation; the
upstream clone is likewise distinct. No correct-forward-output premise is used. -/
theorem gradient_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p))
    (lseLogits : p.lse≠p.logits) (lseUpstream : p.lse≠p.upstream) (lane : Fin p.block) :
    KernelNative.gradientValue algebra p initial lane=
      Source.backwardValues RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1) lane := by
  unfold KernelNative.gradientValue
  rw [logits_scalar p initial abi Bool.true lane]
  cases hs : p.hasSmoothing
  all_goals
    by_cases ignored : KernelNative.target p initial=p.ignored
    all_goals
      simp only [KernelNative.target] at ignored
      simp [KernelNative.inputLSE,KernelNative.inputUpstream,
        Source.backwardValues,Source.arithmetic,Source.scaledReads,CrossEntropy1.Source.backwardValues,hs,ignored,KernelNative.target,CrossEntropy1.Source.label,
        CrossEntropy1.Source.loadedLogits,CrossEntropy1.Source.column,CrossEntropy1.Source.load,CrossEntropy1.Source.logitAddress,
        CrossEntropy1.Source.fp,CrossEntropy1.Source.nat,CrossEntropy1.Source.binary,CrossEntropy1.Source.unary,
        scalarMemory,CrossEntropy1.Source.Cell.read,algebra,CrossEntropy1.KernelReal.algebra,
        RealSource.algebra,CrossEntropy1.RealSource.algebra,RuntimeArgs.fp,RuntimeArgs.classes,
        abi.smoothing,abi.squareScale,abi.totalClasses,abi.logitScale,arguments,lseLogits,lseUpstream]

theorem z_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) :
    KernelNative.zValue algebra p initial=
      (if CrossEntropy1.Source.label p.toParameters (scalarMemory initial) (initial.pids 0)=p.ignored then ↑(0 : ℝ) else
        RealSource.algebra.mul (RealSource.algebra.mul (↑p.squareScale)
          (Source.forwardValues RealSource.algebra p (scalarMemory initial) (initial.pids 0) (initial.pids 1)).lse)
          (Source.forwardValues RealSource.algebra p (scalarMemory initial) (initial.pids 0) (initial.pids 1)).lse) := by
  unfold KernelNative.zValue
  rw [lse_scalar p initial abi]
  simp [target_scalar,algebra,CrossEntropy1.KernelReal.algebra,RuntimeArgs.fp,abi.squareScale,arguments,
    RealSource.algebra,CrossEntropy1.RealSource.algebra]

#axiomsClean logits_scalar
#axiomsClean lse_scalar
#axiomsClean loss_scalar
#axiomsClean gradient_scalar
#axiomsClean z_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelReal

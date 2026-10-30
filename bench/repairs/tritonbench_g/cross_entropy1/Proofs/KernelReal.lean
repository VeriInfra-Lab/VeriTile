import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelBackward
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelMath
import Mathlib.Tactic.NormNum.OfScientific

namespace VeriTile.Bench.Repairs.CrossEntropy1.KernelReal
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

@[simp] theorem decimal_zero : (0.0 : ℝ)=0 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num
@[simp] theorem decimal_one : (1.0 : ℝ)=1 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num
@[simp] theorem decimal_two : (2.0 : ℝ)=2 := by
  rw [← Rat.cast_ofScientific,← Rat.ofScientific_eq_ofScientific,Rat.ofScientific_true_def]
  norm_num

def lanes {α : Type} {shape : TileShape} (ax : Fin shape.length) (keepDims : Bool)
    (xs : Values α .real shape) :
    TileIndex (TileShape.reduceShape shape ax keepDims) → Fin (TileShape.axisDim shape ax) → α :=
  match keepDims with
  | Bool.false => fun i k => xs (TileShape.insertAxisIndex shape ax i k)
  | Bool.true => fun i k => xs (TileShape.replaceAxisIndex shape ax i k)

/-- Ideal real semantics for the selected source operations. Masked negative
infinity is retained as bottom and exp(bottom) is zero. This is a mathematical
interpretation, separate from the opaque native primitives. -/
def algebra : Algebra (WithBot ℝ) where
  literal := fun _ _ x => ↑x
  negInf := ⊥
  binary := fun _ _ op => match op with
    | .add => RealSource.algebra.add
    | .sub => RealSource.algebra.sub
    | .mul => RealSource.algebra.mul
    | .div => RealSource.algebra.div
    | .max => max
    | .pow => fun _ _ => ⊥
  unary := fun _ op => match op with
    | .exp => RealSource.algebra.exp
    | .log => RealSource.algebra.log
    | _ => fun _ => ⊥
  cast := fun _ _ _ => id
  fromNat := fun _ n => ↑(n : ℝ)
  fromInt := fun _ n => ↑(n : ℝ)
  fp32Bits := fun _ => ⊥
  fp32Load := id
  reduceSum := fun _ {shape} ax keepDims xs i => RealSource.algebra.sum (lanes ax keepDims xs i)
  reduceMax := fun _ {shape} ax keepDims xs i => RealSource.algebra.maximum (lanes ax keepDims xs i)

def arguments (p : Parameters) : RuntimeArgs.Values (WithBot ℝ) := ⟨↑p.smoothing,↑p.squareScale⟩

theorem logits_scalar (p : Parameters) (initial : State (WithBot ℝ)) (lane : Fin p.block) :
    KernelNative.logits algebra p initial (lane,PUnit.unit)=
      Source.loadedLogits RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1) lane := by rfl

theorem lse_scalar (p : Parameters) (initial : State (WithBot ℝ)) :
    KernelNative.lseValue algebra p initial=
      (Source.forwardValues RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1)).lse := by rfl

theorem sum_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (hs : p.hasSmoothing=Bool.true) :
    some (KernelNative.sumValue algebra p initial)=
      (Source.forwardValues RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1)).sumLogits := by
  simp [Source.forwardValues,hs,KernelNative.sumValue,KernelNative.sumTile,
    algebra,lanes,RealSource.algebra,Source.sum,Source.fp,← logits_scalar] <;> rfl

@[simp] theorem target_scalar (p : Parameters) (initial : State (WithBot ℝ)) :
    KernelNative.target p initial=Source.label p (scalarMemory initial) (initial.pids 0) := rfl

theorem owns_scalar (p : Parameters) (initial : State (WithBot ℝ)) :
    KernelNative.owns p initial ↔
      ((initial.pids 1*p.block : Nat) : Int)≤KernelNative.target p initial-p.classStart ∧
      KernelNative.target p initial-p.classStart<((min p.cols ((initial.pids 1+1)*p.block) : Nat) : Int) := by
  simp [KernelNative.owns,Nat.cast_mul,Nat.cast_add,Nat.cast_one,Nat.cast_min]

/-- The arithmetic graph proved from actual typed execution equals the
independently analysed scalar formula in the ideal-real interpretation. -/
theorem loss_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) :
    KernelNative.lossValue algebra p initial=
      Source.forwardLoss RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1)
        (Source.label p (scalarMemory initial) (initial.pids 0))
        (Source.forwardValues RealSource.algebra p (scalarMemory initial)
          (initial.pids 0) (initial.pids 1)) := by
  cases hs : p.hasSmoothing <;> cases sp : p.split
  all_goals
    simp [KernelNative.lossValue,KernelNative.coreLoss,Source.forwardLoss,hs,sp,
      ← lse_scalar,KernelNative.owns,KernelNative.target,Source.label,Source.forwardValues,Source.fp,Source.nat,
      Source.binary,Source.load,Source.logitAddress,KernelNative.gathered,
      KernelNative.sumValue,KernelNative.sumTile,Source.sum,
      Source.loadedLogits,Source.column,Source.maximum,Source.unary,
      KernelNative.lseValue,KernelNative.lseTile,KernelNative.maxValue,KernelNative.maxTile,
      KernelNative.logits,scalarMemory,Source.Cell.read,algebra,lanes,RealSource.algebra,
      RuntimeArgs.fp,RuntimeArgs.classes,abi.smoothing,abi.squareScale,abi.totalClasses,arguments] <;> rfl

/-- The same bridge for one backward lane. It contains no execution or
correctness premise and is only arithmetic unfolding. -/
theorem gradient_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) (lane : Fin p.block) :
    KernelNative.gradientValue algebra p initial lane=
      Source.backwardValues RealSource.algebra p (scalarMemory initial)
        (initial.pids 0) (initial.pids 1) lane := by
  cases hs : p.hasSmoothing
  all_goals
    by_cases ignored : KernelNative.target p initial=p.ignored
    all_goals
      simp [KernelNative.gradientValue,KernelNative.inputLSE,KernelNative.inputUpstream,
        Source.backwardValues,hs,ignored,← target_scalar,KernelNative.logits,
        Source.loadedLogits,Source.column,Source.load,Source.logitAddress,
        Source.fp,Source.nat,Source.binary,Source.unary,
        scalarMemory,Source.Cell.read,algebra,RealSource.algebra,
        RuntimeArgs.fp,RuntimeArgs.classes,abi.smoothing,abi.squareScale,abi.totalClasses,arguments]

#axiomsClean lse_scalar
#axiomsClean loss_scalar
#axiomsClean gradient_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy1.KernelReal

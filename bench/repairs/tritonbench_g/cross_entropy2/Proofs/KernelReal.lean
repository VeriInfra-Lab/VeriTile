import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelBackward
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.KernelContract
import bench.repairs.tritonbench_g.cross_entropy2.Proofs.BackwardReal
import bench.repairs.tritonbench_g.cross_entropy1.Proofs.KernelReal

namespace VeriTile.Bench.Repairs.CrossEntropy2.KernelReal
open VeriTile.Triton FP.Structural KernelContract
noncomputable section
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false
abbrev algebra := CrossEntropy1.KernelReal.algebra

def arguments (p : Parameters) : RuntimeArgs.Values (WithBot ℝ) :=
  ⟨⟨↑p.smoothing,↑p.squareScale⟩,↑p.logitScale⟩

theorem logits_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) (lane : Fin p.block) :
    KernelNative.logits algebra p initial (lane,PUnit.unit)=
      CrossEntropy1.Source.loadedLogits (Source.scaled RealSource.algebra p.logitScale) p.toParameters
        (scalarMemory initial) (initial.pids 0) (initial.pids 1) lane := by
  by_cases active : initial.pids 1*p.block+lane.val<p.cols
  all_goals simp [active,KernelNative.logits,Source.scaled,CrossEntropy1.Source.loadedLogits,CrossEntropy1.Source.load,
    CrossEntropy1.Source.column,CrossEntropy1.Source.logitAddress,
    RuntimeArgs.fp,CrossEntropy1.RuntimeArgs.fp,abi.logitScale,arguments,algebra,CrossEntropy1.KernelReal.algebra,
    RealSource.algebra,CrossEntropy1.RealSource.algebra,scalarMemory,CrossEntropy1.Source.Cell.read]

theorem lse_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) :
    KernelNative.lseValue algebra p initial=
      (Source.values RealSource.algebra p (scalarMemory initial) (initial.pids 0) (initial.pids 1)).lse := by
  have loads : CrossEntropy1.Source.loadedLogits (Source.scaled RealSource.algebra p.logitScale) p.toParameters
      (scalarMemory initial) (initial.pids 0) (initial.pids 1)=
        fun lane => KernelNative.logits algebra p initial (lane,PUnit.unit) := by
    funext lane
    exact (logits_scalar p initial abi lane).symm
  simp only [Source.values,CrossEntropy1.Source.forwardValues,loads]
  rfl

theorem sum_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) (enabled : p.hasSmoothing=Bool.true) :
    some (KernelNative.sumValue algebra p initial)=
      (Source.values RealSource.algebra p (scalarMemory initial) (initial.pids 0) (initial.pids 1)).sumLogits := by
  have loads : CrossEntropy1.Source.loadedLogits (Source.scaled RealSource.algebra p.logitScale) p.toParameters
      (scalarMemory initial) (initial.pids 0) (initial.pids 1)=
        fun lane => KernelNative.logits algebra p initial (lane,PUnit.unit) := by
    funext lane
    exact (logits_scalar p initial abi lane).symm
  simp only [Source.values,CrossEntropy1.Source.forwardValues,enabled,↓reduceIte,loads]
  simp [KernelNative.sumValue,KernelNative.sumTile,algebra,CrossEntropy1.KernelReal.algebra,
    CrossEntropy1.KernelReal.lanes,CrossEntropy1.Source.sum,CrossEntropy1.Source.fp,
    CrossEntropy1.Source.column,Source.scaled,RealSource.algebra,CrossEntropy1.RealSource.algebra] <;> rfl

@[simp] theorem target_scalar (p : Parameters) (initial : State (WithBot ℝ)) :
    KernelNative.target p initial=CrossEntropy1.Source.label p.toParameters (scalarMemory initial) (initial.pids 0) := rfl

theorem owns_scalar (p : Parameters) (initial : State (WithBot ℝ)) :
    KernelNative.owns p initial ↔ Source.blockOwns p (KernelNative.target p initial) (initial.pids 1) := by
  simp [KernelNative.owns,Source.blockOwns,Nat.cast_mul,Nat.cast_add,Nat.cast_one,Nat.cast_min]

theorem loss_z_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) :
    (KernelNative.lossValue algebra p initial,KernelNative.zValue algebra p initial)=
      Source.lossZ RealSource.algebra p (scalarMemory initial) (initial.pids 0) (initial.pids 1)
        (KernelNative.target p initial)
        (Source.values RealSource.algebra p (scalarMemory initial) (initial.pids 0) (initial.pids 1)) := by
  by_cases ignored : KernelNative.target p initial=p.ignored
  all_goals cases hs : p.hasSmoothing <;> cases sp : p.split
  all_goals
    simp only [KernelNative.lossValue,KernelNative.zValue,KernelNative.coreLoss,Source.lossZ,
      ignored,hs,sp,Bool.false_eq_true,↓reduceIte,← lse_scalar p initial abi,← owns_scalar p initial]
  all_goals try rw [← sum_scalar p initial abi hs]
  all_goals
    simp [Source.fp,Source.nat,Source.binary,Source.load,
      CrossEntropy1.Source.fp,CrossEntropy1.Source.nat,CrossEntropy1.Source.binary,CrossEntropy1.Source.load,
      KernelNative.owns,Source.blockOwns,Nat.cast_mul,Nat.cast_add,Nat.cast_one,Nat.cast_min,
      CrossEntropy1.Source.logitAddress,
      KernelNative.gathered,scalarMemory,CrossEntropy1.Source.Cell.read,
      algebra,CrossEntropy1.KernelReal.algebra,RealSource.algebra,CrossEntropy1.RealSource.algebra,
      RuntimeArgs.fp,CrossEntropy1.RuntimeArgs.fp,RuntimeArgs.classes,CrossEntropy1.RuntimeArgs.classes,
      abi.smoothing,abi.squareScale,abi.totalClasses,abi.logitScale,arguments]

theorem hit_iff (p : Parameters) (initial : State (WithBot ℝ)) (lane : Fin p.block)
    (active : initial.pids 1*p.block+lane.val<p.cols) :
    (KernelNative.ownsTarget p initial ∧ ((initial.pids 1*p.block+lane.val : Nat) : Int)=
      (if KernelNative.ownsTarget p initial then KernelNative.target p initial else p.classStart)-p.classStart) ↔
      ((initial.pids 1*p.block+lane.val : Nat) : Int)=KernelNative.target p initial-p.classStart := by
  by_cases owner : KernelNative.ownsTarget p initial
  · simp [owner]
  · simp only [owner,if_false,false_and,false_iff]
    unfold KernelNative.ownsTarget at owner
    have lower : (0 : Int)≤(initial.pids 1*p.block+lane.val : Nat) := Int.natCast_nonneg _
    omega

theorem gradient_scalar (p : Parameters) (initial : State (WithBot ℝ))
    (abi : RuntimeArgs.Bound p initial (arguments p)) (lane : Fin p.block)
    (active : initial.pids 1*p.block+lane.val<p.cols) :
    KernelNative.gradientValue algebra p initial lane=
      Source.backwardValues RealSource.algebra p (scalarMemory initial) (initial.pids 0) (initial.pids 1) lane := by
  have hit := hit_iff p initial lane active
  simp only [Nat.cast_add,Nat.cast_mul] at hit
  cases hs : p.hasSmoothing
  all_goals
    by_cases ignored : KernelNative.target p initial=p.ignored
    all_goals
      simp [KernelNative.gradientValue,hit,
        KernelNative.inputLSE,KernelNative.inputUpstream,Source.backwardValues,hs,ignored,
        ← target_scalar,KernelNative.logits,CrossEntropy1.Source.loadedLogits,CrossEntropy1.Source.column,
        Source.scaled,Source.load,CrossEntropy1.Source.logitAddress,
        Source.fp,Source.nat,Source.binary,Source.unary,
        CrossEntropy1.Source.fp,CrossEntropy1.Source.nat,CrossEntropy1.Source.binary,CrossEntropy1.Source.unary,CrossEntropy1.Source.load,
        scalarMemory,CrossEntropy1.Source.Cell.read,algebra,CrossEntropy1.KernelReal.algebra,
        RealSource.algebra,CrossEntropy1.RealSource.algebra,active,
        RuntimeArgs.fp,CrossEntropy1.RuntimeArgs.fp,RuntimeArgs.classes,CrossEntropy1.RuntimeArgs.classes,abi.smoothing,abi.squareScale,abi.totalClasses,abi.logitScale,arguments]

#axiomsClean lse_scalar
#axiomsClean loss_z_scalar
#axiomsClean gradient_scalar
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropy2.KernelReal

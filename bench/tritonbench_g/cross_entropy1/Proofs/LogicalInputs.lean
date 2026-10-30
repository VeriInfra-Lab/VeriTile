import bench.tritonbench_g.cross_entropy1.Proofs.OrderedReference
import bench.tritonbench_g.cross_entropy1.Proofs.KernelBackward

namespace VeriTile.Bench.Repairs.CrossEntropy1.LogicalInputs
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 1000000
variable {α : Type} [Inhabited α]

theorem logits (M : Algebra α) (p : Parameters) (initial : State α)
    (args : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial args)
    (x : Nat → α)
    (binding : ∀ c, c<p.cols →
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+c)).read .real=x c) :
    KernelNative.logits M p initial=Reference.logits M p args x (initial.pids 1) := by
  funext i
  by_cases active : initial.pids 1*p.block+i.1.val<p.cols
  · simp [KernelNative.logits,Reference.logits,active,binding _ active,
      RuntimeArgs.fp]
  · simp [KernelNative.logits,Reference.logits,active]

theorem lse (M : Algebra α) (p : Parameters) (initial : State α)
    (args : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial args)
    (x : Nat → α)
    (binding : ∀ c, c<p.cols →
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+c)).read .real=x c) :
    KernelNative.lseValue M p initial=Reference.lse M p args x (initial.pids 1) := by
  simp only [KernelNative.lseValue,KernelNative.lseTile,KernelNative.maxValue,KernelNative.maxTile,
    Reference.lse,Reference.maximum,logits M p initial args abi x binding]

theorem sum (M : Algebra α) (p : Parameters) (initial : State α)
    (args : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial args)
    (x : Nat → α)
    (binding : ∀ c, c<p.cols →
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+c)).read .real=x c) :
    KernelNative.sumValue M p initial=Reference.sum M p args x (initial.pids 1) := by
  simp only [KernelNative.sumValue,KernelNative.sumTile,Reference.sum,
    logits M p initial args abi x binding]

theorem forward (M : Algebra α) (p : Parameters) (initial : State α)
    (args : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial args)
    (x : Nat → α) (label : Int)
    (binding : ∀ c, c<p.cols →
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+c)).read .real=x c)
    (labelBinding : KernelNative.target p initial=label) :
    (KernelNative.lseValue M p initial,KernelNative.lossValue M p initial)=
      Reference.forward M p args (initial.pids 1) x label := by
  have hl := lse M p initial args abi x binding
  have hs := sum M p initial args abi x binding
  have own : KernelNative.owns p initial ↔ Reference.owns p (initial.pids 1) label := by
    simp only [KernelNative.owns,Reference.owns,labelBinding]
  have gather (hit : Reference.owns p (initial.pids 1) label) :
      KernelNative.gathered M p initial=
        M.fp32Load (x (label-p.classStart).toNat) := by
    have lower : (0 : Int)≤(initial.pids 1 : Int)*p.block := mul_nonneg (Int.natCast_nonneg _) (Int.natCast_nonneg _)
    have upper := lt_of_lt_of_le hit.2 (min_le_left _ _)
    have hk : (label-p.classStart).toNat<p.cols := by
      have nonneg : 0≤label-p.classStart := le_trans lower hit.1
      exact_mod_cast (show (Int.toNat (label-p.classStart) : Int)<p.cols by simpa [Int.toNat_of_nonneg nonneg] using upper)
    simp [KernelNative.gathered,labelBinding,binding _ hk,RuntimeArgs.fp]
  have core : KernelNative.coreLoss M p initial=Reference.coreLoss M p args x (initial.pids 1) label := by
    classical
    by_cases hit : Reference.owns p (initial.pids 1) label
    · simp [KernelNative.coreLoss,Reference.coreLoss,own,hit,gather hit,hl,hs,
        RuntimeArgs.fp,RuntimeArgs.classes,abi.smoothing,abi.totalClasses]
    · simp [KernelNative.coreLoss,Reference.coreLoss,own,hit,hl,hs,
        RuntimeArgs.fp,RuntimeArgs.classes,abi.smoothing,abi.totalClasses]
  by_cases ignored : label=p.ignored
  all_goals cases sp : p.split
  all_goals simp only [Reference.forward,KernelNative.lossValue,Reference.loss,Reference.zLoss,
    labelBinding,hl,core,RuntimeArgs.fp,abi.squareScale,Option.getD_some,ignored,if_true,if_false,sp,Bool.false_eq_true]

theorem backward (M : Algebra α) (p : Parameters) (initial : State α)
    (args : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p initial args)
    (x : Nat → α) (label : Int) (L upstream : α)
    (binding : ∀ c, c<p.cols →
      (initial.mem p.logits (p.logitsBase+initial.pids 0*p.logitsStride+c)).read .real=x c)
    (labelBinding : KernelNative.target p initial=label)
    (lseBinding : (initial.mem p.lse (p.lseBase+initial.pids 0)).read .real=L)
    (upstreamBinding : label≠p.ignored →
      (initial.mem p.upstream (p.upstreamBase+initial.pids 0*p.upstreamStride)).read .real=upstream)
    (lane : Fin p.block) :
    KernelNative.gradientValue M p initial lane=
      Reference.backward M p args (initial.pids 1) x label L upstream lane := by
  by_cases ignored : label=p.ignored
  · simp [KernelNative.gradientValue,Reference.backward,labelBinding,ignored]
  · simp [KernelNative.gradientValue,Reference.backward,KernelNative.inputLSE,KernelNative.inputUpstream,
      labelBinding,ignored,lseBinding,upstreamBinding ignored,
      logits M p initial args abi x binding,RuntimeArgs.fp,RuntimeArgs.classes,
      abi.smoothing,abi.squareScale,abi.totalClasses]

#axiomsClean logits
#axiomsClean lse
#axiomsClean sum
#axiomsClean forward
#axiomsClean backward
end
end VeriTile.Bench.Repairs.CrossEntropy1.LogicalInputs

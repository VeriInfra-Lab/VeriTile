import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelBackward
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.OrderedFormula

/-! Arithmetic locality and typed readback reuse the existing whole-source
execution lemmas. No second interpreter or source proof is introduced. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelNative
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 4000000
variable {A : Type} [Inhabited A]

theorem logits_logical (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) (x : Nat → A)
    (input : ∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + s.pids 0 * p.logitsStride + i)).read .real = x i) :
    logits M p s = Ordered.logits M p v (s.pids 1) x := by
  funext i
  by_cases active : s.pids 1 * p.block + i.1.val < p.cols
  · simp [logits, rawLogits, loadedLogits, Ordered.logits, active, input _ active, RuntimeArgs.fp, abi.logitScale]
  · simp [logits, Ordered.logits, active]

theorem lse_logical (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) (x : Nat → A)
    (input : ∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + s.pids 0 * p.logitsStride + i)).read .real = x i) :
    lseValue M p s = Ordered.lse M p v (s.pids 1) x := by
  simp only [lseValue, lseTile, maxValue, maxTile, logits_logical M p s v abi x input,
    Ordered.lse, Ordered.maximum]

theorem sum_logical (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) (x : Nat → A)
    (input : ∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + s.pids 0 * p.logitsStride + i)).read .real = x i) :
    sumValue M p s = Ordered.sum M p v (s.pids 1) x := by
  simp only [sumValue, sumTile, logits_logical M p s v abi x input, Ordered.sum]

theorem coreLoss_logical (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) (x : Nat → A)
    (input : ∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + s.pids 0 * p.logitsStride + i)).read .real = x i) :
    coreLoss M p s = Ordered.coreLoss M p v (s.pids 1) x (target p s) := by
  classical
  by_cases hit : owns p s
  · have nonnegative : 0 ≤ target p s - p.classStart := by
      have lower := hit.1
      have := mul_nonneg (Int.natCast_nonneg (s.pids 1)) (Int.natCast_nonneg p.block)
      omega
    have inside : (target p s - p.classStart).toNat < p.cols := by
      have upper := lt_of_lt_of_le hit.2 (min_le_left _ _)
      omega
    have hit' : Ordered.owns p (s.pids 1) (target p s) := hit
    simp only [coreLoss, Ordered.coreLoss, hit, hit', if_true,
      lse_logical M p s v abi x input, sum_logical M p s v abi x input,
      gathered, input _ inside, RuntimeArgs.fp, RuntimeArgs.classes,
      abi.smoothing, abi.logitScale, abi.totalClasses, Option.getD_some]
  · have hit' : ¬Ordered.owns p (s.pids 1) (target p s) := hit
    simp only [coreLoss, Ordered.coreLoss, hit, hit', if_false,
      lse_logical M p s v abi x input, sum_logical M p s v abi x input,
      RuntimeArgs.fp, RuntimeArgs.classes, abi.smoothing, abi.totalClasses, Option.getD_some]

theorem loss_logical (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) (x : Nat → A)
    (input : ∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + s.pids 0 * p.logitsStride + i)).read .real = x i) :
    lossValue M p s = Ordered.loss M p v (s.pids 1) x (target p s) := by
  simp only [lossValue, Ordered.loss, coreLoss_logical M p s v abi x input,
    lse_logical M p s v abi x input, RuntimeArgs.fp, abi.squareScale, Option.getD_some]

theorem zloss_logical (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) (x : Nat → A)
    (input : ∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + s.pids 0 * p.logitsStride + i)).read .real = x i) :
    zValue M p s = Ordered.zloss M p v (s.pids 1) x (target p s) := by
  simp only [zValue, Ordered.zloss, lse_logical M p s v abi x input,
    RuntimeArgs.fp, abi.squareScale, Option.getD_some]

theorem gradient_logical (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) (x : Nat → A)
    (input : ∀ i, i < p.cols → (s.mem p.logits (p.logitsBase + s.pids 0 * p.logitsStride + i)).read .real = x i)
    (lane : Fin p.block) :
    gradientValue M p s lane = Ordered.gradient M p v (s.pids 1) x (target p s)
      ((s.mem p.lse (p.lseBase + s.pids 0)).read .real)
      ((s.mem p.upstream (p.upstreamBase + s.pids 0 * p.upstreamStride)).read .real) lane := by
  by_cases ignored : target p s = p.ignored
  all_goals simp only [gradientValue, Ordered.gradient, inputLSE, inputUpstream, ignored,
    if_true, if_false, logits_logical M p s v abi x input, RuntimeArgs.fp,
    RuntimeArgs.classes, abi.smoothing, abi.squareScale, abi.logitScale,
    abi.totalClasses, Option.getD_some]

theorem forward_cells (M : Algebra A) (p : Parameters) (s : State A)
    (positive : 0 < p.block) (separate : p.lse ≠ p.logits) (outputs : p.lse ≠ p.loss)
    (zLse : p.zloss ≠ p.lse) (zLoss : p.zloss ≠ p.loss)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) :
    ∃ final, exec (engine M) (forwardSyntax p) s = some final ∧
      final.mem p.lse (p.lseBase + s.pids 1 * p.rows + s.pids 0) = .mk .real (lseValue M p s) ∧
      final.mem p.loss (p.lossBase + s.pids 1 * p.rows + s.pids 0) = .mk .real (lossValue M p s) ∧
      (p.split = Bool.false → final.mem p.zloss (p.zlossBase + s.pids 1 * p.rows + s.pids 0) = .mk .real (zValue M p s)) ∧
      (∀ region offset,
        (region ≠ p.loss ∨ offset ≠ p.lossBase + s.pids 1 * p.rows + s.pids 0) →
        (region ≠ p.lse ∨ offset ≠ p.lseBase + s.pids 1 * p.rows + s.pids 0) →
        (p.split = Bool.true ∨ region ≠ p.zloss ∨ offset ≠ p.zlossBase + s.pids 1 * p.rows + s.pids 0) →
        final.mem region offset = s.mem region offset) := by
  obtain ⟨final, execution, memory⟩ := forward_executes M p s positive separate v abi
  obtain ⟨other, execution', _, _, _, frame⟩ := forward_outputs M p s positive separate outputs zLse zLoss v abi
  have same : other = final := Option.some.inj (execution'.symm.trans execution)
  subst other
  refine ⟨final, execution, ?_, ?_, ?_, frame⟩
  · rw [memory]
    cases sp : p.split <;> simp [finishMemory, sp, State.write, outputs, zLse.symm]
  · rw [memory]
    cases sp : p.split <;> simp [finishMemory, sp, State.write, zLoss.symm]
  · intro sp
    rw [memory]
    simp [finishMemory, sp, State.write]

theorem backward_cells (M : Algebra A) (p : Parameters) (s : State A)
    (v : RuntimeArgs.Values A) (abi : RuntimeArgs.Bound p s v) :
    ∃ final, exec (engine M) (backwardSyntax p) s = some final ∧
      (∀ lane : Fin p.block, s.pids 1 * p.block + lane.val < p.cols →
        final.mem p.gradient (gradientOffset p s (lane, PUnit.unit)) = .mk .real (gradientValue M p s lane)) ∧
      (∀ region offset,
        (∀ lane : Fin p.block, s.pids 1 * p.block + lane.val < p.cols →
          region ≠ p.gradient ∨ offset ≠ gradientOffset p s (lane, PUnit.unit)) →
        final.mem region offset = s.mem region offset) := by
  obtain ⟨final, execution, memory⟩ := backward_executes M p s v abi
  obtain ⟨other, execution', _, frame⟩ := backward_outputs M p s v abi
  have same : other = final := Option.some.inj (execution'.symm.trans execution)
  subst other
  refine ⟨final, execution, ?_, frame⟩
  intro lane active
  rw [memory]
  exact State.masked_scatter_readback p.gradient (gradientOffset p s)
    (fun i => Cell.mk .real (gradientValue M p s i.1)) (activeLane p s)
    s (lane, PUnit.unit) active (gradientOffset_injective p s)

#axiomsClean forward_cells
#axiomsClean backward_cells
#axiomsClean loss_logical
#axiomsClean gradient_logical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelNative

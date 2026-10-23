import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelBackward
import VeriTile.Triton.Float.StructuralSafety

/-! Actual source accesses, including the conditional gathered label logit,
the ignored-label branch and in-place backward stores. Numerical operations
do not supply or assume any address-safety law. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.KernelNative
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 16000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem forwardPrefix_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (positive : 0 < p.block)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p s values)
    (labels : p.labelsBase + s.pids 0 < bounds (Region.cast p.labels))
    (input : ∀ i, i < p.cols → p.logitsBase + s.pids 0 * p.logitsStride + i < bounds p.logits)
    (output : p.lseBase + s.pids 1 * p.rows + s.pids 0 < bounds p.lse) :
    Safety.traceSafeList (engine M) bounds (front p) s := by
  cases hs : p.hasSmoothing
  all_goals simp (maxSteps := 2200000) [front, forwardSyntax, hs, ComputeKernel.surfaceBody,
    Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
    Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
    Safety.maskActive, run, step, evalExpr, evalComputeOp, evalOp_unfold,
    store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
    numericEq, Region.cast, State.setReg, engine, Algebra.withDefaultPrecision,
    resolvePrecision, TileShape.axisDim, positive, abi.logitScale, ofFloat, toFloat,
    Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals exact ⟨labels, fun i hi => input _ hi, output⟩

theorem forwardTail_safe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (values : RuntimeArgs.Values α)
    (abi : RuntimeArgs.Bound p s values)
    (input : ∀ i, i < p.cols → p.logitsBase + s.pids 0 * p.logitsStride + i < bounds p.logits)
    (loss : p.lossBase + s.pids 1 * p.rows + s.pids 0 < bounds p.loss)
    (zloss : p.split = Bool.false → p.zlossBase + s.pids 1 * p.rows + s.pids 0 < bounds p.zloss) :
    Safety.traceSafeList (engine M) bounds (tailCode p) (ready M p s) := by
  classical
  cases hs : p.hasSmoothing <;> cases sp : p.split
  all_goals
    by_cases ignored : target p s = p.ignored
    · simp (maxSteps := 2200000) [tailCode, forwardSyntax, hs, sp, ComputeKernel.surfaceBody,
        Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
        Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
        Safety.maskActive, run, step, evalExpr, evalComputeOp, evalOp_unfold,
        store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
        numericLe, numericEq, Region.cast, State.setReg, engine, Algebra.withDefaultPrecision,
        resolvePrecision, ofFloat, toFloat, ignored, ready_lse, loss, zloss,
        abi.smoothing, abi.squareScale, abi.totalClasses, abi.logitScale]
    · by_cases hit : owns p s
      all_goals
        have condition := hit
        simp only [owns, lt_min_iff] at condition
        simp (maxSteps := 2200000) [tailCode, forwardSyntax, hs, sp, ComputeKernel.surfaceBody,
          Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr, Safety.safeCompute,
          Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask, Safety.addressSafe,
          Safety.maskActive, run, step, evalExpr, evalComputeOp, evalOp_unfold,
          store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt, numericLt,
          numericLe, numericEq, Region.cast, State.setReg, engine, Algebra.withDefaultPrecision,
          resolvePrecision, ofFloat, toFloat, ignored, ready_lse, condition, loss, zloss,
          abi.smoothing, abi.squareScale, abi.totalClasses, abi.logitScale]
      all_goals
        have nonnegative : 0 ≤ target p s - p.classStart := by
          have := mul_nonneg (Int.natCast_nonneg (s.pids 1)) (Int.natCast_nonneg p.block)
          omega
        have within : (target p s - p.classStart).toNat < p.cols := by omega
        simpa only [Int.toNat_add (by positivity) nonnegative,
          Int.toNat_natCast] using input _ within

theorem forward_traceSafe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (positive : 0 < p.block)
    (values : RuntimeArgs.Values α) (abi : RuntimeArgs.Bound p s values)
    (labels : p.labelsBase + s.pids 0 < bounds (Region.cast p.labels))
    (input : ∀ i, i < p.cols → p.logitsBase + s.pids 0 * p.logitsStride + i < bounds p.logits)
    (lse : p.lseBase + s.pids 1 * p.rows + s.pids 0 < bounds p.lse)
    (loss : p.lossBase + s.pids 1 * p.rows + s.pids 0 < bounds p.loss)
    (zloss : p.split = Bool.false → p.zlossBase + s.pids 1 * p.rows + s.pids 0 < bounds p.zloss) :
    FP.Structural.TraceSafe (engine M) bounds (forwardSyntax p) s := by
  rw [FP.Structural.TraceSafe, body_split, Safety.traceSafeList_append]
  refine ⟨forwardPrefix_safe M p s bounds positive values abi labels input lse, ?_⟩
  intro t ht
  rw [forwardPrefix M p s positive values abi] at ht
  cases ht
  exact forwardTail_safe M p s bounds values abi input loss zloss

theorem backward_traceSafe (M : Algebra α) (p : Parameters) (s : State α)
    (bounds : RegionName → Nat) (values : RuntimeArgs.Values α)
    (abi : RuntimeArgs.Bound p s values)
    (labels : p.labelsBase + s.pids 0 < bounds (Region.cast p.labels))
    (input : ∀ i, i < p.cols → p.logitsBase + s.pids 0 * p.logitsStride + i < bounds p.logits)
    (lse : p.lseBase + s.pids 0 < bounds p.lse)
    (upstream : target p s ≠ p.ignored → p.upstreamBase + s.pids 0 * p.upstreamStride < bounds p.upstream)
    (gradient : ∀ i : Fin p.block, s.pids 1 * p.block + i.val < p.cols →
      p.gradientBase + s.pids 0 * p.gradientStride + (s.pids 1 * p.block + i.val) < bounds p.gradient) :
    FP.Structural.TraceSafe (engine M) bounds (backwardSyntax p) s := by
  cases hs : p.hasSmoothing
  all_goals by_cases ignored : target p s = p.ignored
  all_goals simp only [target, Region.cast] at ignored upstream
  all_goals simp (maxSteps := 2200000) [FP.Structural.TraceSafe, backwardSyntax, hs,
    ComputeKernel.surfaceBody, Safety.traceSafeList, Safety.stepSafe, Safety.safeExpr,
    Safety.safeCompute, Safety.safeOp.eq_def, Safety.safeAddress, Safety.safeMask,
    Safety.addressSafe, Safety.maskActive, run, step, evalExpr, evalComputeOp,
    evalOp_unfold, store, Option.bind, ComputeDType.eraseDType, numeric, bop, natLt,
    numericLt, numericLe, numericEq, Region.cast, State.setReg, engine,
    Algebra.withDefaultPrecision, resolvePrecision, ofFloat, toFloat, ignored,
    labels, lse, abi.smoothing, abi.squareScale, abi.totalClasses, abi.logitScale,
    Broadcast.leftIndex, Broadcast.rightIndex]
  all_goals repeat' constructor
  all_goals first
    | exact labels
    | exact upstream ignored
    | exact lse
    | exact gradient
    | exact fun i hi => input _ hi

#axiomsClean forward_traceSafe
#axiomsClean backward_traceSafe
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CrossEntropyOps.KernelNative

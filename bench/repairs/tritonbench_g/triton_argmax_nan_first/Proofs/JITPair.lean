import bench.repairs.tritonbench_g.triton_argmax_nan_first.CandidateKernels
import VeriTile.Triton.Float.ProgramIO
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITPair
open VeriTile Triton FP.Structural
noncomputable section
set_option linter.unusedSimpArgs false

namespace First
def before (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :=
  (JITSource.stage1 X V I dtype block wide).surfaceBody.take 2
def after (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :=
  (JITSource.stage1 X V I dtype block wide).surfaceBody.drop 3
def originalSite (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :=
  ((JITSource.stage1 X V I dtype block wide).surfaceBody[2]?).getD (.effectMarker "unreachable")
def candidateSite (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :=
  ((JITCandidate.stage1 X V I dtype block wide).surfaceBody[2]?).getD (.effectMarker "unreachable")

theorem original_code (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :
    (JITSource.stage1 X V I dtype block wide).surfaceBody =
      before X V I dtype block wide ++ originalSite X V I dtype block wide :: after X V I dtype block wide := by
  cases dtype <;> rfl

theorem candidate_code (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :
    (JITCandidate.stage1 X V I dtype block wide).surfaceBody =
      before X V I dtype block wide ++ candidateSite X V I dtype block wide :: after X V I dtype block wide := by
  cases dtype <;> rfl

theorem site_run {α : Type} [Inhabited α] (A : Algebra α) (s : State α)
    (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :
    step A (originalSite X V I dtype block wide) s = step A (candidateSite X V I dtype block wide) s := by
  cases dtype <;>
    simp only [originalSite, candidateSite, JITSource.stage1, JITCandidate.stage1,
      ComputeKernel.surfaceBody, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some]
  all_goals by_cases h : block = 1024
  all_goals simp [h, step, run, evalExpr, evalComputeOp, evalOp_unfold,
    Op.intShiftLeft, IntegerExpr.sameShape, numeric, bop, Broadcast.leftIndex,
    Broadcast.rightIndex, Int.natCast_pow, Nat.shiftLeft_eq]

theorem kernel_run {α : Type} [Inhabited α] (A : Algebra α) (s : State α)
    (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) :
    exec A (JITSource.stage1 X V I dtype block wide) s =
      exec A (JITCandidate.stage1 X V I dtype block wide) s := by
  change run A (JITSource.stage1 X V I dtype block wide).surfaceBody s =
    run A (JITCandidate.stage1 X V I dtype block wide).surfaceBody s
  rw [original_code, candidate_code, run_append, run_append]
  apply congrArg (fun f => (run A (before X V I dtype block wide) s).bind f)
  funext t
  simp only [run, site_run]

theorem site_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat)
    (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) (s : State α) :
    Safety.stepSafe A bounds (originalSite X V I dtype block wide) s ∧
    Safety.stepSafe A bounds (candidateSite X V I dtype block wide) s := by
  cases dtype <;>
    simp only [originalSite, candidateSite, JITSource.stage1, JITCandidate.stage1,
      ComputeKernel.surfaceBody, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some]
  all_goals by_cases h : block = 1024
  all_goals simp [h, Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeOp_unfold,
    Safety.traceSafeList_cons, Safety.traceSafeList_nil, evalOp_unfold, Op.intShiftLeft,
    IntegerExpr.sameShape]

theorem kernel_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat)
    (X V : RegionName) (I : Region .int) (dtype : SourceDType) (block : Nat) (wide : Bool) (s : State α) :
    TraceSafe A bounds (JITSource.stage1 X V I dtype block wide) s ↔
    TraceSafe A bounds (JITCandidate.stage1 X V I dtype block wide) s := by
  unfold TraceSafe
  rw [original_code, candidate_code, Safety.traceSafeList_append, Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons, Safety.traceSafeList_cons]
  simp only [(site_safe A bounds X V I dtype block wide t).1,
    (site_safe A bounds X V I dtype block wide t).2, true_and, site_run]

#axiomsClean kernel_run
#axiomsClean kernel_safe
end First

namespace Dimension
def before (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :=
  (JITSource.dimension X I block columns wide).surfaceBody.take 4
def after (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :=
  (JITSource.dimension X I block columns wide).surfaceBody.drop 5
def originalSite (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :=
  ((JITSource.dimension X I block columns wide).surfaceBody[4]?).getD (.effectMarker "unreachable")
def candidateSite (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :=
  ((JITCandidate.dimension X I block columns wide).surfaceBody[4]?).getD (.effectMarker "unreachable")

theorem original_code (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :
    (JITSource.dimension X I block columns wide).surfaceBody =
      before X I block columns wide ++ originalSite X I block columns wide :: after X I block columns wide := by
  rfl

theorem candidate_code (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :
    (JITCandidate.dimension X I block columns wide).surfaceBody =
      before X I block columns wide ++ candidateSite X I block columns wide :: after X I block columns wide := by
  rfl

theorem site_run {α : Type} [Inhabited α] (A : Algebra α) (s : State α)
    (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :
    step A (originalSite X I block columns wide) s = step A (candidateSite X I block columns wide) s := by
  simp only [originalSite, candidateSite, JITSource.dimension, JITCandidate.dimension,
      ComputeKernel.surfaceBody, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some]
  all_goals by_cases h : block = 128
  all_goals simp [h, step, run, evalExpr, evalComputeOp, evalOp_unfold,
    Op.intShiftLeft, IntegerExpr.sameShape, numeric, bop, Broadcast.leftIndex,
    Broadcast.rightIndex, Int.natCast_pow, Nat.shiftLeft_eq]

theorem kernel_run {α : Type} [Inhabited α] (A : Algebra α) (s : State α)
    (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) :
    exec A (JITSource.dimension X I block columns wide) s =
      exec A (JITCandidate.dimension X I block columns wide) s := by
  change run A (JITSource.dimension X I block columns wide).surfaceBody s =
    run A (JITCandidate.dimension X I block columns wide).surfaceBody s
  rw [original_code, candidate_code, run_append, run_append]
  apply congrArg (fun f => (run A (before X I block columns wide) s).bind f)
  funext t
  simp only [run, site_run]

theorem site_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat)
    (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) (s : State α) :
    Safety.stepSafe A bounds (originalSite X I block columns wide) s ∧
    Safety.stepSafe A bounds (candidateSite X I block columns wide) s := by
  simp only [originalSite, candidateSite, JITSource.dimension, JITCandidate.dimension,
      ComputeKernel.surfaceBody, List.getElem?_cons_succ, List.getElem?_cons_zero, Option.getD_some]
  all_goals by_cases h : block = 128
  all_goals simp [h, Safety.stepSafe_unfold, Safety.safeExpr_unfold, Safety.safeOp_unfold,
    Safety.traceSafeList_cons, Safety.traceSafeList_nil, evalOp_unfold, Op.intShiftLeft,
    IntegerExpr.sameShape]

theorem kernel_safe {α : Type} [Inhabited α] (A : Algebra α) (bounds : RegionName → Nat)
    (X : RegionName) (I : Region .int) (block columns : Nat) (wide : Bool) (s : State α) :
    TraceSafe A bounds (JITSource.dimension X I block columns wide) s ↔
    TraceSafe A bounds (JITCandidate.dimension X I block columns wide) s := by
  unfold TraceSafe
  rw [original_code, candidate_code, Safety.traceSafeList_append, Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons, Safety.traceSafeList_cons]
  simp only [(site_safe A bounds X I block columns wide t).1,
    (site_safe A bounds X I block columns wide t).2, true_and, site_run]

#axiomsClean kernel_run
#axiomsClean kernel_safe
end Dimension

end
end VeriTile.Bench.Repairs.TritonArgmaxNanFirst.JITPair

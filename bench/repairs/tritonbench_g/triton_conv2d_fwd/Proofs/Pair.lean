import bench.repairs.tritonbench_g.triton_conv2d_fwd.IO
import bench.repairs.tritonbench_g.triton_conv2d_fwd.Proofs.PublicResult
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.Conv2d.Pair
open VeriTile Triton FP.Structural KernelNumeric
noncomputable section
set_option maxHeartbeats 400000
set_option linter.unusedSimpArgs false
set_option maxRecDepth 32000

def before (s : Shape) (f : Flags) := (originalKernel s f).surfaceBody.take 5
def after (s : Shape) (f : Flags) := (originalKernel s f).surfaceBody.drop 6
def originalSite (s : Shape) (f : Flags) :=
  ((originalKernel s f).surfaceBody[5]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (s : Shape) (f : Flags) :=
  ((kernel s f).surfaceBody[6]? : Option ComputeStmt).getD (.effectMarker "unreachable")

/-- The source static assertion is true at every existing public invocation.
Its retained no-runtime-work statement is proved safe and inert below. -/
def compileGuard : ComputeStmt := .ifThen (.constBool Bool.false) []

theorem public_block_size (s : Shape) (f : Flags) :
    (KernelBinding.parameters s f).BLOCK_SIZE_BATCH_HEIGHT_WIDTH = 128 := rfl

theorem guard_run {α : Type} [Inhabited α] (M : Algebra α) (s : State α) :
    step M compileGuard s = some s := by
  simp [compileGuard,step,evalOp_unfold]

theorem guard_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (s : State α) : Safety.stepSafe M bounds compileGuard s := by
  simp [compileGuard,Safety.stepSafe_unfold,Safety.safeOp_unfold,evalOp_unfold]

theorem original_code (s : Shape) (f : Flags) :
    (originalKernel s f).surfaceBody = before s f ++ originalSite s f :: after s f := by
  rcases f with ⟨f,t⟩
  cases f <;> rfl

theorem shifted_code (s : Shape) (f : Flags) :
    (kernel s f).surfaceBody = compileGuard :: (before s f ++ shiftedSite s f :: after s f) := by
  rcases f with ⟨f,t⟩
  cases f <;> rfl

theorem site_run {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (f : Flags) (s : State α) :
    step M (originalSite c f) s = step M (shiftedSite c f) s := by
  rcases f with ⟨f,t⟩
  cases f
  all_goals simp only [originalSite,shiftedSite,originalKernel,kernel,KernelSource.kernel,CandidateSource.kernel,
    KernelHalf.source,KernelDefault.source,CandidateHalf.source,CandidateDefault.source,
    KernelHalf.halfBody,KernelDefault.defaultBody,CandidateHalf.halfBody,CandidateDefault.defaultBody,KernelBinding.parameters,
    Bool.false_eq_true,ite_false,ite_true,ComputeKernel.surfaceBody,
    List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  all_goals simp only [step,evalExpr,evalOp_unfold]
  all_goals cases hr : s.regs .nat [] "batch_height_width_pid"
  all_goals simp [hr,numeric,FP.Structural.bop,Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals unfold FP.Structural.bop
  all_goals simp [Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex]

theorem site_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (c : Shape) (f : Flags) (s : State α) :
    Safety.stepSafe M bounds (originalSite c f) s ∧ Safety.stepSafe M bounds (shiftedSite c f) s := by
  rcases f with ⟨f,t⟩
  cases f
  all_goals simp only [originalSite,shiftedSite,originalKernel,kernel,KernelSource.kernel,CandidateSource.kernel,
    KernelHalf.source,KernelDefault.source,CandidateHalf.source,CandidateDefault.source,
    KernelHalf.halfBody,KernelDefault.defaultBody,CandidateHalf.halfBody,CandidateDefault.defaultBody,KernelBinding.parameters,
    Bool.false_eq_true,ite_false,ite_true,ComputeKernel.surfaceBody,
    List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  all_goals simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,and_true,true_and]

theorem run_cons {α : Type} [Inhabited α] (M : Algebra α)
    (st : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    run M (st::rest) s = (step M st s).bind (run M rest) := by
  rw [run]
  rfl

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (c : Shape) (f : Flags) (s : State α) :
    exec M (originalKernel c f) s = exec M (kernel c f) s := by
  have er (k : ComputeKernel) : exec M k s = run M k.surfaceBody s := by cases k; rfl
  rw [er,er,original_code,shifted_code,run_cons,guard_run]
  simp only [Option.bind_some]
  rw [run_append,run_append]
  apply congrArg (fun g => (run M (before c f) s).bind g)
  funext t
  rw [run_cons,run_cons,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (c : Shape) (f : Flags) (s : State α) :
    TraceSafe M bounds (originalKernel c f) s ↔ TraceSafe M bounds (kernel c f) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_cons]
  simp only [guard_safe,guard_run,true_and,Option.some.injEq,forall_eq']
  rw [Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe M bounds c f t).1,(site_safe M bounds c f t).2,true_and,site_run]

theorem invocation_run {A : Type} [Inhabited A] (o : Operations A)
    (s : Shape) (f : Flags) (pid : Program) (t : State (ConvIO.Payload A)) :
    (ConvIO.invocation o s f pid).run t = (invocation o s f pid).run t := by
  change exec (KernelGrid.backend (algebra o) (halfDot o) (defaultDot o) f) (originalKernel s f)
    ((ConvIO.invocation o s f pid).initial t) =
      exec (KernelGrid.backend (algebra o) (halfDot o) (defaultDot o) f) (kernel s f)
        ((ConvIO.invocation o s f pid).initial t)
  exact kernel_run _ s f _

theorem invocation_safe {A : Type} [Inhabited A] (o : Operations A) (bounds : RegionName → Nat)
    (s : Shape) (f : Flags) (pid : Program) (t : State (ConvIO.Payload A)) :
    (ConvIO.invocation o s f pid).Safe bounds t ↔ (invocation o s f pid).Safe bounds t := by
  change TraceSafe (KernelGrid.backend (algebra o) (halfDot o) (defaultDot o) f) bounds (originalKernel s f)
    ((ConvIO.invocation o s f pid).initial t) ↔
      TraceSafe (KernelGrid.backend (algebra o) (halfDot o) (defaultDot o) f) bounds (kernel s f)
        ((ConvIO.invocation o s f pid).initial t)
  exact kernel_safe _ bounds s f _

theorem schedule_run {A : Type} [Inhabited A] (o : Operations A)
    (s : Shape) (f : Flags) (work : List Program) (t : State (ConvIO.Payload A)) :
    runSchedule (work.map (ConvIO.invocation o s f)) t = runSchedule (work.map (invocation o s f)) t := by
  induction work generalizing t with
  | nil => rfl
  | cons job rest ih =>
    simp only [List.map_cons,runSchedule,←invocation_run]
    cases he : (ConvIO.invocation o s f job).run t with
    | none => rfl
    | some next => exact ih next

theorem schedule_safe {A : Type} [Inhabited A] (o : Operations A) (bounds : RegionName → Nat)
    (s : Shape) (f : Flags) (work : List Program) (t : State (ConvIO.Payload A)) :
    ScheduleSafe bounds (work.map (ConvIO.invocation o s f)) t ↔
      ScheduleSafe bounds (work.map (invocation o s f)) t := by
  induction work generalizing t with
  | nil => rfl
  | cons job rest ih =>
    simp only [List.map_cons,ScheduleSafe,←invocation_safe,←invocation_run]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)

theorem equivalent {A : Type} [Inhabited A] (o : Operations A) (s : Shape) (v : Valid s) (f : Flags) :
    (ConvIO.program o s f).Equivalent (variant o s f) := by
  intro vals _ bounds backed initial pinned
  obtain ⟨safe,final,ran,outputs,frame⟩ := ConvIO.implements o s v f vals trivial bounds backed initial pinned
  refine ⟨safe,(schedule_safe o bounds s f _ initial).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule ((programs s).map (invocation o s f)) initial = some final
    rw [←schedule_run]
    exact ran
  · intro k j _
    rfl

#axiomsClean public_block_size
#axiomsClean guard_run
#axiomsClean guard_safe
#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.Conv2d.Pair

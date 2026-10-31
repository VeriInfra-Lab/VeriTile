import bench.tritonbench_g.attention_fwd_triton2.Kernels
import bench.tritonbench_g.attention_fwd_triton2.Proofs.IOResults
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.AttentionFwdTriton2.Pair
open VeriTile Triton FP.Structural Whole001 Source Whole Machine Interface
noncomputable section
set_option maxHeartbeats 300000
set_option maxRecDepth 32000

def before (p : Parameters) := (Source.kernel p).surfaceBody.take 10
def after (p : Parameters) := (Source.kernel p).surfaceBody.drop 11
def originalSite (p : Parameters) :=
  ((Source.kernel p).surfaceBody[10]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (p : Parameters) :=
  ((kernel p).surfaceBody[10]? : Option ComputeStmt).getD (.effectMarker "unreachable")

theorem original_code (p : Parameters) :
    (Source.kernel p).surfaceBody = before p ++ originalSite p :: after p := rfl

theorem shifted_code (p : Parameters) :
    (kernel p).surfaceBody = before p ++ shiftedSite p :: after p := rfl

theorem site_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    step M (originalSite p) s = step M (shiftedSite p) s := by
  simp only [originalSite,shiftedSite,Source.kernel,kernel,TypedSource.repaired,CandidateSource.shifted,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  simp only [step,evalExpr,evalOp_unfold]
  cases hr : s.regs .int [] "start_m"
  all_goals simp [hr,numeric,FP.Structural.bop,Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex]
  all_goals unfold FP.Structural.bop
  all_goals simp [Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex]

theorem site_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (p : Parameters) (s : State α) :
    Safety.stepSafe M bounds (originalSite p) s ∧ Safety.stepSafe M bounds (shiftedSite p) s := by
  simp only [originalSite,shiftedSite,Source.kernel,kernel,TypedSource.repaired,CandidateSource.shifted,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  simp only [Safety.stepSafe_unfold,Safety.safeExpr_unfold,Safety.safeOp_unfold,and_true,true_and]

theorem run_cons {α : Type} [Inhabited α] (M : Algebra α)
    (st : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    run M (st::rest) s = (step M st s).bind (run M rest) := by
  rw [run]
  rfl

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    exec M (Source.kernel p) s = exec M (kernel p) s := by
  change run M (Source.kernel p).surfaceBody s = run M (kernel p).surfaceBody s
  rw [original_code,shifted_code,run_append,run_append]
  apply congrArg (fun f => (run M (before p) s).bind f)
  funext t
  rw [run_cons,run_cons,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (p : Parameters) (s : State α) :
    TraceSafe M bounds (Source.kernel p) s ↔ TraceSafe M bounds (kernel p) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe M bounds p t).1,(site_safe M bounds p t).2,true_and,site_run]

theorem invocation_run {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (p : Parameters) (program : Program) (s : State α) :
    (Interface.invocation M dot p program).run s = (invocation M dot p program).run s := by
  change exec (FP.ProfiledRewrite.engine (withDot M dot)) (Source.kernel p)
    ((Interface.invocation M dot p program).initial s) =
      exec (FP.ProfiledRewrite.engine (withDot M dot)) (kernel p)
        ((Interface.invocation M dot p program).initial s)
  exact kernel_run _ p _

theorem invocation_safe {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (bounds : RegionName → Nat) (p : Parameters) (program : Program) (s : State α) :
    (Interface.invocation M dot p program).Safe bounds s ↔ (invocation M dot p program).Safe bounds s := by
  change TraceSafe (FP.ProfiledRewrite.engine (withDot M dot)) bounds (Source.kernel p)
    ((Interface.invocation M dot p program).initial s) ↔
      TraceSafe (FP.ProfiledRewrite.engine (withDot M dot)) bounds (kernel p)
        ((Interface.invocation M dot p program).initial s)
  exact kernel_safe _ bounds p _

theorem schedule_run {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (p : Parameters) (work : List Program) (s : State α) :
    runSchedule (work.map (Interface.invocation M dot p)) s =
      runSchedule (work.map (invocation M dot p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons job rest ih =>
    simp only [List.map_cons,runSchedule,←invocation_run]
    cases he : (Interface.invocation M dot p job).run s with
    | none => rfl
    | some t => exact ih t

theorem schedule_safe {α : Type} [Inhabited α] (M : Algebra α) (dot : Dot α)
    (bounds : RegionName → Nat) (p : Parameters) (work : List Program) (s : State α) :
    ScheduleSafe bounds (work.map (Interface.invocation M dot p)) s ↔
      ScheduleSafe bounds (work.map (invocation M dot p)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons job rest ih =>
    simp only [List.map_cons,ScheduleSafe,←invocation_safe,←invocation_run]
    exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ => ih t)

theorem equivalent {α β : Type} [Inhabited α] [Inhabited β]
    (M : Algebra β) (dot : Dot β) (p : Parameters) (bases : Bases) (dom : Domain p bases)
    (sep : p.Out≠p.Q ∧ p.Out≠p.K ∧ p.Out≠p.V ∧ p.Out≠p.QS ∧ p.Out≠p.KS)
    (encode : α → β) (observe : TileIndex [p.batches,p.heads,p.length,96] → β → α)
    (work : List Program) (order : (programs (grid p)).Perm work) :
    (io M dot p encode observe work).Equivalent (variant M dot p encode observe work) := by
  intro vals _ bounds backed initial pinned
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    IOResults.implements M dot p bases dom sep encode observe work order vals trivial bounds backed initial pinned
  refine ⟨safe,(schedule_safe M dot bounds p work initial).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule (work.map (invocation M dot p)) initial = some final
    rw [←schedule_run]
    exact ran
  · intro o j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.Repairs.AttentionFwdTriton2.Pair

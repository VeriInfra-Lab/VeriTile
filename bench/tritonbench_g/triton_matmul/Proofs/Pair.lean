import bench.tritonbench_g.triton_matmul.PairIO
import bench.tritonbench_g.triton_matmul.CandidateKernels
import bench.tritonbench_g.triton_matmul.Proofs.PublicIO
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Pair
open VeriTile Triton FP.Structural FPValue
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 32000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def before (p : Parameters) (st : Storage) := (Source.typed p st).surfaceBody.take 3
def after (p : Parameters) (st : Storage) := (Source.typed p st).surfaceBody.drop 4
def originalSite (p : Parameters) (st : Storage) :=
  ((Source.typed p st).surfaceBody[3]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def shiftedSite (p : Parameters) (st : Storage) :=
  ((CandidateSource.typed p st).surfaceBody[3]? : Option ComputeStmt).getD (.effectMarker "unreachable")

theorem original_code (p : Parameters) (st : Storage) :
    (Source.typed p st).surfaceBody = before p st ++ originalSite p st :: after p st := rfl

theorem shifted_code (p : Parameters) (st : Storage) :
    (CandidateSource.typed p st).surfaceBody = before p st ++ shiftedSite p st :: after p st := rfl

theorem site_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) :
    step M (originalSite p st) s = step M (shiftedSite p st) s := by
  simp only [originalSite,shiftedSite,Source.typed,CandidateSource.typed,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : p.GROUP_SIZE_M=8
  all_goals simp only [hg,decide_true,decide_false]
  all_goals simp only [step, FP.Structural.run, evalExpr, evalOp_unfold]
  all_goals cases hr : s.regs .nat [] "num_pid_n"
  all_goals simp [FP.Structural.run,step,evalExpr,evalOp_unfold,hr,hg,numeric,FP.Structural.bop,
    Nat.shiftLeft_eq,Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]
  all_goals unfold FP.Structural.bop
  all_goals congr 1
  all_goals funext i
  all_goals simp [Broadcast.leftIndex,Broadcast.rightIndex,Nat.mul_comm]

theorem site_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (p : Parameters) (st : Storage) (s : State α) :
    Safety.stepSafe M bounds (originalSite p st) s ∧
      Safety.stepSafe M bounds (shiftedSite p st) s := by
  simp only [originalSite,shiftedSite,Source.typed,CandidateSource.typed,
    ComputeKernel.surfaceBody,List.getElem?_cons_succ,List.getElem?_cons_zero,Option.getD_some]
  by_cases hg : p.GROUP_SIZE_M=8
  all_goals simp only [hg,decide_true,decide_false,Safety.stepSafe_unfold,Safety.safeExpr_unfold,
    Safety.safeOp_unfold,Safety.traceSafeList_cons,Safety.traceSafeList_nil,evalOp_unfold,
    implies_true,and_true,true_and,Bool.false_eq_true,if_true,if_false]

theorem run_cons {α : Type} [Inhabited α] (M : Algebra α)
    (st : ComputeStmt) (rest : List ComputeStmt) (s : State α) :
    FP.Structural.run M (st::rest) s = (step M st s).bind (FP.Structural.run M rest) := by rw [FP.Structural.run]; rfl

theorem kernel_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (st : Storage) (s : State α) :
    exec M (Source.typed p st) s = exec M (CandidateSource.typed p st) s := by
  change FP.Structural.run M (Source.typed p st).surfaceBody s = FP.Structural.run M (CandidateSource.typed p st).surfaceBody s
  rw [original_code,shifted_code,run_append,run_append]
  apply congrArg (fun f => (FP.Structural.run M (before p st) s).bind f)
  funext t
  rw [run_cons,run_cons,site_run]

theorem kernel_safe {α : Type} [Inhabited α] (M : Algebra α) (bounds : RegionName → Nat)
    (p : Parameters) (st : Storage) (s : State α) :
    TraceSafe M bounds (Source.typed p st) s ↔ TraceSafe M bounds (CandidateSource.typed p st) s := by
  unfold TraceSafe
  rw [original_code,shifted_code,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons]
  simp only [(site_safe M bounds p st t).1,(site_safe M bounds p st t).2,true_and,site_run]

theorem invocation_run {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (pid : Nat) (s : State α) :
    (PublicIO.invocation g M D pid).run s = (invocation g M D pid).run s := by
  change exec (backend M D) (Source.typed g.request.parameters g.storage) (instanceState g.request pid s) =
    exec (backend M D) (CandidateSource.typed g.request.parameters g.storage) (instanceState g.request pid s)
  exact kernel_run _ _ _ _

theorem invocation_safe {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (bounds : RegionName → Nat) (pid : Nat) (s : State α) :
    (PublicIO.invocation g M D pid).Safe bounds s ↔ (invocation g M D pid).Safe bounds s := by
  change TraceSafe (backend M D) bounds (Source.typed g.request.parameters g.storage) (instanceState g.request pid s) ↔
    TraceSafe (backend M D) bounds (CandidateSource.typed g.request.parameters g.storage) (instanceState g.request pid s)
  exact kernel_safe _ _ _ _ _

theorem schedule_run {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (ids : List Nat) (s : State α) :
    runSchedule (ids.map (PublicIO.invocation g M D)) s = runSchedule (ids.map (invocation g M D)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid ids ih =>
    simp only [List.map_cons,runSchedule,←invocation_run]
    cases he : (PublicIO.invocation g M D pid).run s with
    | none => rfl
    | some next => exact ih next

theorem schedule_safe {α : Type} [Inhabited α] (g : Geometry) (M : Algebra α) (D : DotAcc α)
    (bounds : RegionName → Nat) (ids : List Nat) (s : State α) :
    ScheduleSafe bounds (ids.map (PublicIO.invocation g M D)) s ↔
      ScheduleSafe bounds (ids.map (invocation g M D)) s := by
  induction ids generalizing s with
  | nil => rfl
  | cons pid ids ih =>
    simp only [List.map_cons,ScheduleSafe,←invocation_safe,←invocation_run]
    exact and_congr Iff.rfl (forall_congr' fun next => imp_congr_right fun _ => ih next)

theorem equivalent (rest : Algebra Word) (I : Native.Instructions) (g : Geometry) (ids : List Nat)
    (order : ids.Perm (List.range (programs g.request))) (sepA : g.C≠g.A) (sepB : g.C≠g.B) :
    (io rest I g ids).Equivalent (variant rest I g ids) := by
  intro vals _ bounds backed initial pinned
  have proved := PublicIO.implements g (Native.algebra rest I) (Native.dotAcc I)
    (Native.numerics I g.storage) (Native.primitives rest I g.storage) ids order sepA sepB
    (logicalReference (Native.numerics I g.storage) g.request)
    (reference_logical (Native.numerics I g.storage) g.request)
  obtain ⟨safe,final,ran,outputs,frame⟩ := proved vals trivial bounds backed initial pinned
  refine ⟨safe,(schedule_safe g _ _ bounds ids initial).mp safe,final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule (ids.map (invocation g (Native.algebra rest I) (Native.dotAcc I))) initial=some final
    rw [←schedule_run]
    exact ran
  · intro k j _
    rfl

#axiomsClean equivalent
end
end VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.Pair

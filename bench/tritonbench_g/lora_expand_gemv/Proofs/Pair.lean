import bench.tritonbench_g.lora_expand_gemv.PairIO
import bench.tritonbench_g.lora_expand_gemv.CandidateKernels
import bench.tritonbench_g.lora_expand_gemv.Proofs.IOCorrect
import VeriTile.Triton.Float.ProgramEquivalenceSyntax

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Pair
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

/-- The readable selected source preserves the complete old macro expansion. -/
theorem original_source (p : Branches.Parameters) :
    fullSource p=Branches.source .singleHalf .single p := rfl

def active (k : ComputeKernel) : List ComputeStmt :=
  match (k.surfaceBody[3]? : Option ComputeStmt) with
  | some (ComputeStmt.ifThen _ body) => body
  | _ => []
def before (g : Geometry) := (active (fullSource g.parameters)).take 3
def originalSite (g : Geometry) : ComputeStmt :=
  ((active (fullSource g.parameters))[3]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def candidateSite (g : Geometry) : ComputeStmt :=
  ((active (unitSplitSource g.parameters))[3]? : Option ComputeStmt).getD (.effectMarker "unreachable")
def after (g : Geometry) := (active (fullSource g.parameters)).drop 4

theorem original_split (g : Geometry) :
    active (fullSource g.parameters)=before g++originalSite g::after g := rfl
theorem candidate_split (g : Geometry) :
    active (unitSplitSource g.parameters)=before g++candidateSite g::after g := rfl

theorem site_run (A : Algebra Word) (g : Geometry) (positive : 0<g.N) (s : State Word) :
    step A (originalSite g) s=step A (candidateSite g) s := by
  have one := length_one g positive
  change (g.N+g.N-1)/g.N=1 at one
  simp [originalSite,candidateSite,active,fullSource,unitSplitSource,
    Geometry.parameters,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource,step,evalExpr,evalComputeOp,
    evalOp_unfold,Option.bind,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,one]
  unfold bop
  simp [Broadcast.leftIndex,Broadcast.rightIndex,one]

theorem site_safe (A : Algebra Word) (g : Geometry) (bounds : RegionName → Nat) (s : State Word) :
    Safety.stepSafe A bounds (originalSite g) s ↔ Safety.stepSafe A bounds (candidateSite g) s := by
  simp [originalSite,candidateSite,active,fullSource,unitSplitSource,
    Geometry.parameters,ComputeKernel.surfaceBody,VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.fullSource,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold,Safety.safeOp_unfold]


theorem run_cons (A : Algebra Word) (st : ComputeStmt) (rest : List ComputeStmt) (s : State Word) :
    run A (st::rest) s=(step A st s).bind (run A rest) := by rw [run]; rfl

theorem active_run (A : Algebra Word) (g : Geometry) (positive : 0<g.N) (s : State Word) :
    run A (active (fullSource g.parameters)) s=run A (active (unitSplitSource g.parameters)) s := by
  rw [original_split,candidate_split,run_append,run_append]
  apply congrArg (fun f => (run A (before g) s).bind f)
  funext t
  rw [run_cons,run_cons,site_run A g positive]

theorem active_safe (A : Algebra Word) (g : Geometry) (positive : 0<g.N)
    (bounds : RegionName → Nat) (s : State Word) :
    Safety.traceSafeList A bounds (active (fullSource g.parameters)) s ↔
      Safety.traceSafeList A bounds (active (unitSplitSource g.parameters)) s := by
  rw [original_split,candidate_split,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  rw [Safety.traceSafeList_cons,Safety.traceSafeList_cons,site_run A g positive,
    site_safe A g bounds t]

def header (g : Geometry) := (fullSource g.parameters).surfaceBody.take 3
def condition (g : Geometry) : Op .bool [] :=
  match ((fullSource g.parameters).surfaceBody[3]? : Option ComputeStmt) with
  | some (ComputeStmt.ifThen cond _) => cond
  | _ => Op.constBool Bool.false

theorem original_code (g : Geometry) : (fullSource g.parameters).surfaceBody=
    header g++[.ifThen (condition g) (active (fullSource g.parameters))] := rfl
theorem candidate_code (g : Geometry) : (unitSplitSource g.parameters).surfaceBody=
    header g++[.ifThen (condition g) (active (unitSplitSource g.parameters))] := rfl

theorem kernel_run (A : Algebra Word) (g : Geometry) (positive : 0<g.N) (s : State Word) :
    exec A (fullSource g.parameters) s=exec A (unitSplitSource g.parameters) s := by
  change run A (fullSource g.parameters).surfaceBody s=run A (unitSplitSource g.parameters).surfaceBody s
  rw [original_code,candidate_code,run_append,run_append]
  apply congrArg (fun f => (run A (header g) s).bind f)
  funext t
  simp only [run_cons,step,active_run A g positive]

theorem kernel_safe (A : Algebra Word) (g : Geometry) (positive : 0<g.N)
    (bounds : RegionName → Nat) (s : State Word) :
    TraceSafe A bounds (fullSource g.parameters) s ↔ TraceSafe A bounds (unitSplitSource g.parameters) s := by
  unfold TraceSafe
  rw [original_code,candidate_code,Safety.traceSafeList_append,Safety.traceSafeList_append]
  apply and_congr_right
  intro _
  apply forall_congr'
  intro t
  apply imp_congr_right
  intro _
  simp only [Safety.traceSafeList_cons,Safety.traceSafeList_nil,and_true]
  simp [Safety.stepSafe_unfold,active_safe A g positive bounds]


theorem invocation_run (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (q : Nat) (bounded : q<g.count) (s : State Word) :
    (launch rest plans g q).run s=(invocation rest plans g q).run s := by
  have positive : 0<g.N := by have := (coordinates g q bounded).1; omega
  change exec (Branches.engine rest .ieeeValue plans) (fullSource g.parameters)
    ((launch rest plans g q).initial s)=
      exec (Branches.engine rest .ieeeValue plans) (unitSplitSource g.parameters)
        ((launch rest plans g q).initial s)
  exact kernel_run _ g positive _

theorem invocation_safe (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (q : Nat) (bounded : q<g.count) (bounds : RegionName → Nat) (s : State Word) :
    (launch rest plans g q).Safe bounds s ↔ (invocation rest plans g q).Safe bounds s := by
  have positive : 0<g.N := by have := (coordinates g q bounded).1; omega
  change TraceSafe (Branches.engine rest .ieeeValue plans) bounds (fullSource g.parameters)
    ((launch rest plans g q).initial s) ↔
      TraceSafe (Branches.engine rest .ieeeValue plans) bounds (unitSplitSource g.parameters)
        ((launch rest plans g q).initial s)
  exact kernel_safe _ g positive bounds _

theorem schedule_run (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (work : List Nat) (bounded : ∀ q∈work,q<g.count) (s : State Word) :
    runSchedule (work.map (launch rest plans g)) s=
      runSchedule (work.map (invocation rest plans g)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.map_cons,runSchedule,←invocation_run rest plans g q (bounded q (by simp))]
    cases he : (launch rest plans g q).run s with
    | none => rfl
    | some t => exact ih (fun p hp => bounded p (by simp [hp])) t

theorem schedule_safe (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (work : List Nat) (bounded : ∀ q∈work,q<g.count)
    (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (work.map (launch rest plans g)) s ↔
      ScheduleSafe bounds (work.map (invocation rest plans g)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons q qs ih =>
    simp only [List.map_cons,ScheduleSafe,
      ←invocation_safe rest plans g q (bounded q (by simp)),
      ←invocation_run rest plans g q (bounded q (by simp))]
    exact and_congr Iff.rfl (forall_congr' fun t => imp_congr_right fun _ =>
      ih (fun p hp => bounded p (by simp [hp])) t)

/-- Both concrete complete grids are safe and successful, all output cells
agree, and both exterior frames hold. Even empty grids retain their full scope. -/
theorem equivalent (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (raw : Nat → Int) (valid : Valid g raw) (order : List Nat)
    (coverage : order.Perm (List.range g.count)) :
    (io rest plans g order).EquivalentOnCells (variant rest plans g order) (domain g raw) := by
  intro vals hd bounds hb initial hin
  obtain ⟨safe,final,ran,outputs,frame⟩ :=
    IOCorrect.graph rest plans g raw valid order coverage vals hd bounds hb initial hin
  have bounded : ∀ q∈order,q<g.count :=
    fun q hq => List.mem_range.mp (coverage.mem_iff.mp hq)
  refine ⟨safe,(schedule_safe rest plans g order bounded bounds initial).mp safe,
    final,final,ran,?_,?_,frame,frame⟩
  · change runSchedule (order.map (invocation rest plans g)) initial=some final
    rw [←schedule_run rest plans g order bounded]
    exact ran
  · intro o j _
    rfl


/-- Existing memory projection of the actual candidate kernel, with the same
full source programs and their original program-id/num-program bindings. -/
def executeUnitSplit (rest : Algebra Word) (plans : ∀ n,ReductionPlan n) (g : Geometry) :
    List Nat → Memory → Option Memory
  | [],mem => some mem
  | q::qs,mem => (exec (Branches.engine rest .ieeeValue plans)
      (unitSplitSource g.parameters) (program g (q/g.N) (q%g.N) mem)).bind
        fun next => executeUnitSplit rest plans g qs next.mem

theorem execute_same (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (order : List Nat) (bounded : ∀ q∈order,q<g.count) (mem : Memory) :
    executeUnitSplit rest plans g order mem=execute rest plans g order mem := by
  induction order generalizing mem with
  | nil => rfl
  | cons q qs ih =>
    have positive : 0<g.N := by
      have := (coordinates g q (bounded q (by simp))).1
      omega
    simp only [executeUnitSplit,execute]
    change (exec (Branches.engine rest .ieeeValue plans) (unitSplitSource g.parameters)
      (program g (q/g.N) (q%g.N) mem)).bind _ =
        (exec (Branches.engine rest .ieeeValue plans) (fullSource g.parameters)
          (program g (q/g.N) (q%g.N) mem)).bind _
    rw [kernel_run _ g positive]
    apply congrArg (fun f => (exec (Branches.engine rest .ieeeValue plans)
      (unitSplitSource g.parameters) (program g (q/g.N) (q%g.N) mem)).bind f)
    funext next
    exact ih (fun p hp => bounded p (by simp [hp])) next.mem

#axiomsClean original_source
#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent
#axiomsClean execute_same

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Pair

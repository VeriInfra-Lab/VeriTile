import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.SourceConnection
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.PairIO
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.CandidateKernels
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IOCorrect
import VeriTile.Triton.Float.ProgramEquivalenceSyntax
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.IntegerCandidateSafety
import bench.repairs.tritonbench_g.lora_expand_gemv.ByteSeparated.Proofs.LocalObservation

namespace VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Pair
open VeriTile.Triton FP.Structural FPValue LoraIO
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 8000000
set_option maxRecDepth 40000
set_option linter.unusedSimpArgs false

/-- The actual pasted source retains the complete historical execution under
its existing graph interpretation, including all branches and final registers. -/
theorem original_source (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (p : Branches.Parameters) (s : State Word) :
    exec (Branches.engine rest .ieeeValue plans) (fullSource p) s=
      exec (Branches.engine rest .ieeeValue plans) (Branches.source .singleHalf .single p) s :=
  SourceConnection.run_eq rest .ieeeValue plans p s

/-- Both admitted signed index-storage profiles retain the complete candidate
body. Native width/range obligations are still carried by the caller. -/
theorem candidate_index_profiles (p : Branches.Parameters) :
    unitSplitSourceInt32 p=unitSplitSource p := rfl

open IntegerCandidate (forgetLocals)

/-- Local integer carriers may differ. Every other state component, including
memory and undefined-load behavior, agrees after actual source execution. -/
theorem kernel_run (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry) (positive : 0<g.N) (s : State Word) :
    (exec (Branches.engine rest .ieeeValue plans) (fullSource g.parameters) s).map forgetLocals=
      (exec (Branches.engine rest .ieeeValue plans) (unitSplitSource g.parameters) s).map forgetLocals :=
  IntegerCandidate.observed_run rest plans g positive s

theorem kernel_safe (rest : Algebra Word) (plans : ∀ n, ReductionPlan n) (g : Geometry) (positive : 0<g.N)
    (bounds : RegionName → Nat) (s : State Word) :
    TraceSafe (Branches.engine rest .ieeeValue plans) bounds (fullSource g.parameters) s ↔ TraceSafe (Branches.engine rest .ieeeValue plans) bounds (unitSplitSource g.parameters) s :=
  IntegerCandidate.safe_iff rest plans g positive bounds s

private theorem matched_result {a b : Option (State Word)}
    (same : a.map forgetLocals=b.map forgetLocals) {s : State Word} (ran : a=some s) :
    ∃ t,b=some t ∧ forgetLocals s=forgetLocals t := by
  have hb : b.map forgetLocals=some (forgetLocals s) := by rw [←same,ran];rfl
  obtain ⟨t,ht,hs⟩ := Option.map_eq_some_iff.mp hb
  exact ⟨t,ht,hs.symm⟩

theorem invocation_run (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (q : Nat) (bounded : q<g.count) (s : State Word) :
    ((launch rest plans g q).run s).map forgetLocals=
      ((invocation rest plans g q).run s).map forgetLocals := by
  have positive : 0<g.N := by have := (coordinates g q bounded).1; omega
  change (exec (Branches.engine rest .ieeeValue plans) (fullSource g.parameters)
    ((launch rest plans g q).initial s)).map forgetLocals=
      (exec (Branches.engine rest .ieeeValue plans) (unitSplitSource g.parameters)
        ((launch rest plans g q).initial s)).map forgetLocals
  exact kernel_run rest plans g positive _

theorem invocation_safe (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (q : Nat) (bounded : q<g.count) (bounds : RegionName → Nat) (s : State Word) :
    (launch rest plans g q).Safe bounds s ↔ (invocation rest plans g q).Safe bounds s := by
  have positive : 0<g.N := by have := (coordinates g q bounded).1; omega
  change TraceSafe (Branches.engine rest .ieeeValue plans) bounds (fullSource g.parameters)
    ((launch rest plans g q).initial s) ↔
      TraceSafe (Branches.engine rest .ieeeValue plans) bounds (unitSplitSource g.parameters)
        ((launch rest plans g q).initial s)
  exact kernel_safe rest plans g positive bounds _

private theorem clear_original (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (work : List Nat) :
    ∀ job∈work.map (launch rest plans g),job.clearRegisters=Bool.true := by
  intro job member
  obtain ⟨q,_,rfl⟩ := List.mem_map.mp member
  rfl

private theorem clear_candidate (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (work : List Nat) :
    ∀ job∈work.map (invocation rest plans g),job.clearRegisters=Bool.true := by
  intro job member
  obtain ⟨q,_,rfl⟩ := List.mem_map.mp member
  rfl

theorem schedule_run (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (work : List Nat) (bounded : ∀ q∈work,q<g.count) (s : State Word) :
    (runSchedule (work.map (launch rest plans g)) s).map forgetLocals=
      (runSchedule (work.map (invocation rest plans g)) s).map forgetLocals := by
  induction work generalizing s with
  | nil => rfl
  | cons q qs ih =>
    have heads := invocation_run rest plans g q (bounded q (by simp)) s
    have tail : ∀ p∈qs,p<g.count := fun p hp => bounded p (by simp [hp])
    simp only [List.map_cons,runSchedule]
    cases left : (launch rest plans g q).run s with
    | none =>
      have right : (invocation rest plans g q).run s=none := by
        apply Option.map_eq_none_iff.mp
        rw [←heads,left];rfl
      simp [left,right]
    | some a =>
      obtain ⟨b,right,same⟩ := matched_result heads left
      simp only [left,right,Option.bind_some]
      exact (ih tail a).trans (IntegerCandidate.schedule_run_congr _
        (clear_candidate rest plans g qs) a b same)

theorem schedule_safe (rest : Algebra Word) (plans : ∀ n,ReductionPlan n)
    (g : Geometry) (work : List Nat) (bounded : ∀ q∈work,q<g.count)
    (bounds : RegionName → Nat) (s : State Word) :
    ScheduleSafe bounds (work.map (launch rest plans g)) s ↔
      ScheduleSafe bounds (work.map (invocation rest plans g)) s := by
  induction work generalizing s with
  | nil => rfl
  | cons q qs ih =>
    have tail : ∀ p∈qs,p<g.count := fun p hp => bounded p (by simp [hp])
    have heads := invocation_run rest plans g q (bounded q (by simp)) s
    have safe := invocation_safe rest plans g q (bounded q (by simp)) bounds s
    simp only [List.map_cons,ScheduleSafe]
    constructor
    · rintro ⟨first,restSafe⟩
      refine ⟨safe.mp first,?_⟩
      intro b right
      obtain ⟨a,left,same⟩ := matched_result heads.symm right
      have next := (ih tail a).mp (restSafe a left)
      exact (IntegerCandidate.schedule_safe_congr _ (clear_candidate rest plans g qs)
        bounds a b same.symm).mp next
    · rintro ⟨first,restSafe⟩
      refine ⟨safe.mpr first,?_⟩
      intro a left
      obtain ⟨b,right,same⟩ := matched_result heads left
      have next := (ih tail b).mpr (restSafe b right)
      exact (IntegerCandidate.schedule_safe_congr _ (clear_original rest plans g qs)
        bounds b a same.symm).mp next

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
  obtain ⟨other,otherRun,same⟩ := matched_result
    (schedule_run rest plans g order bounded initial) ran
  have memory : final.mem=other.mem := congrArg State.mem same
  refine ⟨safe,(schedule_safe rest plans g order bounded bounds initial).mp safe,
    final,other,ran,otherRun,?_,frame,?_⟩
  · intro o j _
    change (final.mem _ _).read _=(other.mem _ _).read _
    rw [memory]
  · simpa only [←memory] using frame


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
        (exec (Branches.engine rest .ieeeValue plans) (Branches.source .singleHalf .single g.parameters)
          (program g (q/g.N) (q%g.N) mem)).bind _
    rw [←SourceConnection.run_eq]
    have same := kernel_run rest plans g positive (program g (q/g.N) (q%g.N) mem)
    cases left : exec (Branches.engine rest .ieeeValue plans) (fullSource g.parameters)
        (program g (q/g.N) (q%g.N) mem) with
    | none =>
      have right : exec (Branches.engine rest .ieeeValue plans) (unitSplitSource g.parameters)
          (program g (q/g.N) (q%g.N) mem)=none := by
        apply Option.map_eq_none_iff.mp
        rw [←same,left];rfl
      simp [left,right]
    | some a =>
      obtain ⟨b,right,observed⟩ := matched_result same left
      simp only [left,right,Option.bind_some]
      have memory : a.mem=b.mem := congrArg State.mem observed
      rw [←memory]
      exact ih (fun p hp => bounded p (by simp [hp])) a.mem

#axiomsClean candidate_index_profiles
#axiomsClean original_source
#axiomsClean kernel_run
#axiomsClean kernel_safe
#axiomsClean equivalent
#axiomsClean execute_same

end
end VeriTile.Bench.Repairs.LoraExpandGemv.ByteSeparated.Pair

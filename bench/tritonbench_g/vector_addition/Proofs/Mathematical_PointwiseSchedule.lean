import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedSchedule

/-! A reusable whole-program scheduling argument for pointwise source
executions. Every application must supply an actual single-program execution
theorem and same-offset locality proof. No source correctness is inferred
from this abstraction alone. Logical cells and whole-program permutations do
not model native byte aliasing or instruction interleaving. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters gridCount grid_covers)
noncomputable section

abbrev Memory (α : Type) := RegionName → Nat → Cell α
abbrev Result (α : Type) := Memory α → Nat → Cell α

def Pointwise {α : Type} (value : Result α) : Prop :=
  ∀ a b o, (∀ r, a r o = b r o) → value a o = value b o

def oneMemory {α : Type} (value : Result α) (p : Parameters)
    (pid : Nat) (mem : Memory α) : Memory α :=
  fun r o => if r = p.output ∧ pid*p.block ≤ o ∧
      o < pid*p.block+p.block ∧ o < p.n then value mem o else mem r o

def scheduledMemory {α : Type} (value : Result α) (p : Parameters)
    (initial : Memory α) (done : List Nat) : Memory α :=
  fun r o => if r = p.output ∧ o < p.n ∧ o / p.block ∈ done then
    value initial o else initial r o

def finalMemory {α : Type} (value : Result α) (p : Parameters)
    (initial : Memory α) : Memory α :=
  fun r o => if r = p.output ∧ o < p.n then value initial o else initial r o

theorem fresh_inputs {α : Type} (value : Result α) (locality : Pointwise value)
    (p : Parameters) (initial : Memory α) (done : List Nat) (pid o : Nat)
    (fresh : pid ∉ done) (owner : o / p.block = pid) :
    value (scheduledMemory value p initial done) o = value initial o := by
  apply locality
  intro r
  simp [scheduledMemory, owner, fresh]

theorem reference_step {α : Type} (value : Result α) (locality : Pointwise value)
    (p : Parameters) (initial : Memory α) (positive : 0 < p.block)
    (done : List Nat) (pid : Nat) (fresh : pid ∉ done) :
    oneMemory value p pid (scheduledMemory value p initial done) =
      scheduledMemory value p initial (done ++ [pid]) := by
  funext r o
  unfold oneMemory
  have window := TypedFloat.program_window_iff p positive pid o
  by_cases region : r = p.output
  · by_cases active : o < p.n
    · by_cases owner : o / p.block = pid
      · have bounds := window.mpr owner
        rw [if_pos ⟨region, bounds.1, bounds.2, active⟩]
        rw [fresh_inputs value locality p initial done pid o fresh owner]
        simp [scheduledMemory, region, active, owner]
      · have outside : ¬ (r = p.output ∧ pid*p.block ≤ o ∧
            o < pid*p.block+p.block ∧ o < p.n) := by
          rintro ⟨_, lo, hi, _⟩
          exact owner (window.mp ⟨lo, hi⟩)
        rw [if_neg outside]
        simp [scheduledMemory, region, active, owner]
    · simp [active, scheduledMemory]
  · simp [region, scheduledMemory]

def runSchedule {α : Type} (execute : Nat → State α → Option (State α)) :
    List Nat → State α → Option (State α)
  | [], current => some current
  | pid :: jobs, current => (execute pid current).bind (runSchedule execute jobs)

theorem schedule_aux_correct {α : Type} (value : Result α) (locality : Pointwise value)
    (p : Parameters) (execute : Nat → State α → Option (State α))
    (step : ∀ pid current, ∃ final, execute pid current = some final ∧
      final.mem = oneMemory value p pid current.mem)
    (initial : Memory α) (positive : 0 < p.block) (jobs done : List Nat)
    (current : State α) (unique : (done ++ jobs).Nodup)
    (memory : current.mem = scheduledMemory value p initial done) :
    ∃ final, runSchedule execute jobs current = some final ∧
      final.mem = scheduledMemory value p initial (done ++ jobs) := by
  induction jobs generalizing done current with
  | nil => exact ⟨current, rfl, by simpa using memory⟩
  | cons pid jobs ih =>
    have fresh : pid ∉ done := by
      intro member
      exact (List.nodup_append.mp unique).2.2 pid member pid (List.mem_cons_self) rfl
    obtain ⟨next, he, hm⟩ := step pid current
    have nextMemory : next.mem = scheduledMemory value p initial (done ++ [pid]) := by
      rw [hm, memory, reference_step value locality p initial positive done pid fresh]
    have nextUnique : ((done ++ [pid]) ++ jobs).Nodup := by
      simpa [List.append_assoc] using unique
    obtain ⟨final, hr, hf⟩ := ih (done ++ [pid]) next nextUnique nextMemory
    refine ⟨final, ?_, ?_⟩
    · simp only [runSchedule, he, Option.bind_some]
      exact hr
    · simpa [List.append_assoc] using hf

theorem complete_memory {α : Type} (value : Result α) (p : Parameters)
    (initial : Memory α) (positive : 0 < p.block) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridCount p))) :
    scheduledMemory value p initial jobs = finalMemory value p initial := by
  funext r o
  by_cases active : o < p.n
  · have below : o < gridCount p * p.block := lt_of_lt_of_le active (grid_covers p positive)
    have quotient : o / p.block < gridCount p := (Nat.div_lt_iff_lt_mul positive).mpr below
    have member : o / p.block ∈ jobs := order.mem_iff.mpr (List.mem_range.mpr quotient)
    simp [scheduledMemory, finalMemory, active, member]
  · simp [scheduledMemory, finalMemory, active]

theorem complete_schedule {α : Type} (value : Result α) (locality : Pointwise value)
    (p : Parameters) (execute : Nat → State α → Option (State α))
    (step : ∀ pid current, ∃ final, execute pid current = some final ∧
      final.mem = oneMemory value p pid current.mem)
    (initial : State α) (positive : 0 < p.block) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, runSchedule execute jobs initial = some final ∧
      final.mem = finalMemory value p initial.mem := by
  have unique : ([] ++ jobs).Nodup := by
    simpa using (order.nodup_iff.mpr (List.nodup_range))
  have memory : initial.mem = scheduledMemory value p initial.mem [] := by
    funext r o
    simp [scheduledMemory]
  obtain ⟨final, he, hm⟩ := schedule_aux_correct value locality p execute step
    initial.mem positive jobs [] initial unique memory
  refine ⟨final, he, ?_⟩
  simpa only [List.nil_append] using hm.trans (by
    simpa only [List.nil_append] using complete_memory value p initial.mem positive jobs order)

#specNonCircular finalMemory avoiding [runSchedule]
#specNonCircular scheduledMemory avoiding [runSchedule]
#axiomsClean fresh_inputs
#axiomsClean reference_step
#axiomsClean schedule_aux_correct
#axiomsClean complete_memory
#axiomsClean complete_schedule
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms

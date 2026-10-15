import bench.tritonbench_g.vector_addition.Proofs.TypedGrid

/-! Any complete permutation of whole typed floating programs has the same
initial-memory result. Programs execute on preceding actual memory. Flat
element offsets partition the reads and writes even for equal logical region
names. This is not native byte aliasing or instruction-level concurrency. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters gridCount programState grid_covers)
noncomputable section

/-- A closed partial-grid formula, using membership rather than execution. -/
def scheduledMemory {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α)
    (done : List Nat) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.n ∧ o / p.block ∈ done then
    resultCell M dx dy dc p initial.mem o else initial.mem r o

theorem program_window_iff (p : Parameters) (positive : 0 < p.block)
    (pid o : Nat) :
    (pid * p.block ≤ o ∧ o < pid * p.block + p.block) ↔ o / p.block = pid := by
  constructor
  · rintro ⟨lo, hi⟩
    apply Nat.div_eq_of_lt_le lo
    simpa [Nat.add_mul] using hi
  · intro owner
    have decomp := Nat.mod_add_div o p.block
    have bound := Nat.mod_lt o positive
    rw [owner, Nat.mul_comm p.block pid] at decomp
    omega

theorem scheduled_fresh_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α)
    (done : List Nat) (pid o : Nat) (fresh : pid ∉ done)
    (owner : o / p.block = pid) :
    resultCell M dx dy dc p (scheduledMemory M dx dy dc p initial done) o =
      resultCell M dx dy dc p initial.mem o := by
  simp [resultCell, scheduledMemory, owner, fresh]

theorem scheduled_reference_step {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) (done : List Nat) (pid : Nat) (fresh : pid ∉ done) :
    expectedMemory M dx dy dc p
      (programState p initial (scheduledMemory M dx dy dc p initial done) pid) =
      scheduledMemory M dx dy dc p initial (done ++ [pid]) := by
  funext r o
  rw [expected_cell]
  simp only [programState, ↓reduceIte]
  have window := program_window_iff p positive pid o
  by_cases region : r = p.output
  · by_cases active : o < p.n
    · by_cases owner : o / p.block = pid
      · have bounds := window.mpr owner
        rw [if_pos ⟨region, bounds.1, bounds.2, active⟩]
        rw [scheduled_fresh_inputs M dx dy dc p initial done pid o fresh owner]
        simp [scheduledMemory, region, active, owner]
      · have outside : ¬ (r = p.output ∧ pid*p.block ≤ o ∧
            o < pid*p.block+p.block ∧ o < p.n) := by
          rintro ⟨_, lo, hi, _⟩
          exact owner (window.mp ⟨lo, hi⟩)
        rw [if_neg outside]
        simp [scheduledMemory, region, active, owner]
    · simp [active, scheduledMemory]
  · simp [region, scheduledMemory]

/-- The list is executed in its given order, resetting local registers and
retaining the actual preceding memory for each complete program. -/
def runScheduleAux {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α) :
    List Nat → State α → Option (State α)
  | [], current => some current
  | pid :: jobs, current =>
      (FP.Structural.exec (castBackend M) (materialized dx dy dc p)
        (programState p initial current.mem pid)).bind
          (runScheduleAux M dx dy dc p initial jobs)

def runSchedule {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α) (jobs : List Nat) :=
  runScheduleAux M dx dy dc p initial jobs initial

theorem schedule_aux_correct {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) (jobs done : List Nat) (current : State α)
    (unique : (done ++ jobs).Nodup)
    (memory : current.mem = scheduledMemory M dx dy dc p initial done) :
    ∃ final, runScheduleAux M dx dy dc p initial jobs current = some final ∧
      final.mem = scheduledMemory M dx dy dc p initial (done ++ jobs) := by
  induction jobs generalizing done current with
  | nil => exact ⟨current, rfl, by simpa using memory⟩
  | cons pid jobs ih =>
    have fresh : pid ∉ done := by
      intro member
      exact (List.nodup_append.mp unique).2.2 pid member pid (List.mem_cons_self) rfl
    obtain ⟨next, he, hm⟩ := complete_correct M dx dy dc p
      (programState p initial current.mem pid)
    have nextMemory : next.mem = scheduledMemory M dx dy dc p initial (done ++ [pid]) := by
      rw [hm, memory, scheduled_reference_step M dx dy dc p initial positive done pid fresh]
    have nextUnique : ((done ++ [pid]) ++ jobs).Nodup := by
      simpa [List.append_assoc] using unique
    obtain ⟨final, hr, hf⟩ := ih (done ++ [pid]) next nextUnique nextMemory
    refine ⟨final, ?_, ?_⟩
    · simp only [runScheduleAux, he, Option.bind_some]
      exact hr
    · simpa [List.append_assoc] using hf

theorem schedule_correct {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) (jobs : List Nat) (unique : jobs.Nodup) :
    ∃ final, runSchedule M dx dy dc p initial jobs = some final ∧
      final.mem = scheduledMemory M dx dy dc p initial jobs := by
  apply schedule_aux_correct M dx dy dc p initial positive jobs [] initial
  · simpa using unique
  · funext r o
    simp [scheduledMemory]

theorem complete_scheduled_memory {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridCount p))) :
    scheduledMemory M dx dy dc p initial jobs = gridExpectedMemory M dx dy dc p initial := by
  funext r o
  by_cases active : o < p.n
  · have below : o < gridCount p * p.block := lt_of_lt_of_le active (grid_covers p positive)
    have quotient : o / p.block < gridCount p := (Nat.div_lt_iff_lt_mul positive).mpr below
    have member : o / p.block ∈ jobs := order.mem_iff.mpr (List.mem_range.mpr quotient)
    simp [scheduledMemory, gridExpectedMemory, active, member]
  · simp [scheduledMemory, gridExpectedMemory, active]

theorem complete_schedule_correct {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α)
    (positive : 0 < p.block) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, runSchedule M dx dy dc p initial jobs = some final ∧
      final.mem = gridExpectedMemory M dx dy dc p initial := by
  obtain ⟨final, he, hm⟩ := schedule_correct M dx dy dc p initial positive jobs
    (order.nodup_iff.mpr (List.nodup_range))
  exact ⟨final, he, hm.trans (complete_scheduled_memory M dx dy dc p initial positive jobs order)⟩

#specNonCircular scheduledMemory avoiding [runScheduleAux, runSchedule, materialized,
  source, FP.Structural.exec]
#axiomsClean program_window_iff
#axiomsClean scheduled_fresh_inputs
#axiomsClean scheduled_reference_step
#axiomsClean schedule_aux_correct
#axiomsClean schedule_correct
#axiomsClean complete_scheduled_memory
#axiomsClean complete_schedule_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat

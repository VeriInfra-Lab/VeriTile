import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostScheduleInputs
import bench.tritonbench_g.batched_vecmat_mult.Proofs.HostReadback

/-! Arbitrary schedules of complete actual programs. The reference performs
only initial-input reads, preserving a fixed reduction plan and the original
rounded block-accumulation order inside every program. Distinct program writes
commute because their actual output addresses differ, not because floating
addition or arbitrary reduction plans are assumed equivalent. -/
namespace VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs
open VeriTile.Triton FP.Structural
open FP.Equational (ReductionPlan)
noncomputable section
set_option maxHeartbeats 3000000
local instance scheduleWord : Inhabited Word := ⟨FP.Value.zero .fp32⟩

/-- A single program's fixed initial-input reference is overlaid only on its
exact source write set. Other cells retain the supplied preceding memory. -/
def applyInitialProgram (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (pid : Nat) (previous : Memory) : Memory := by
  classical
  exact fun r o => if ProgramWrites p initial pid r o then
    expectedMemory cfg plans p (programState p initial initial.mem pid) r o else previous r o

theorem program_expected_memory_from_initial (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (mem : Memory) (pid : Nat)
    (valid : pid < gridCount p) (ha : mem p.A = initial.mem p.A)
    (hb : mem p.B = initial.mem p.B) :
    expectedMemory cfg plans p (programState p initial mem pid) =
      applyInitialProgram cfg plans p initial pid mem := by
  classical
  funext r o
  by_cases write : ProgramWrites p initial pid r o
  · simp only [applyInitialProgram, write, ↓reduceIte]
    rcases write with ⟨rfl, i, rfl⟩
    change expectedMemory cfg plans p (programState p initial mem pid) p.output
      (outputOffset p (programState p initial mem pid) i) =
      expectedMemory cfg plans p (programState p initial initial.mem pid) p.output
        (outputOffset p (programState p initial initial.mem pid) i)
    rw [actual_program_expected_readback cfg plans p initial mem pid valid i,
      actual_program_expected_readback cfg plans p initial initial.mem pid valid i]
    congr 1
    exact accumulated_input_regions_congr cfg plans p
      (programState p initial mem pid) (programState p initial initial.mem pid)
      rfl ha hb (p.K / p.BK) (by rfl) i
  · have outside : Outside p (programState p initial mem pid) r o := by
      by_cases region : r = p.output
      · right
        intro i same
        exact write ⟨region, i, same⟩
      · exact Or.inl region
    rw [expectedMemory_frame cfg plans p (programState p initial mem pid) r o outside]
    simp only [applyInitialProgram, write, ↓reduceIte, programState]

theorem initial_program_preserves_inputs (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (separate : InputsSeparate p)
    (pid : Nat) (mem : Memory) :
    applyInitialProgram cfg plans p initial pid mem p.A = mem p.A ∧
      applyInitialProgram cfg plans p initial pid mem p.B = mem p.B := by
  classical
  constructor <;> funext o
  · simp only [applyInitialProgram, ProgramWrites, separate.1, false_and, ↓reduceIte]
  · simp only [applyInitialProgram, ProgramWrites, separate.2, false_and, ↓reduceIte]

/-- Complete program effects commute, including repeated IDs. The distinct-ID
case uses derived address uniqueness over the real floor-divided host grid. -/
theorem initial_programs_commute (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (mem : Memory)
    (k l : Nat) (hk : k < gridCount p) (hl : l < gridCount p) :
    applyInitialProgram cfg plans p initial l (applyInitialProgram cfg plans p initial k mem) =
      applyInitialProgram cfg plans p initial k (applyInitialProgram cfg plans p initial l mem) := by
  classical
  by_cases same : k = l
  · subst l
    rfl
  funext r o
  have disjoint := distinct_program_write_sets p initial k l hk hl same r o
  simp only [applyInitialProgram]
  by_cases left : ProgramWrites p initial k r o <;>
    by_cases right : ProgramWrites p initial l r o <;> simp_all

/-- Reference for arbitrary partial schedules, reading the original input
memory for every program. No actual execution appears in this definition. -/
def scheduledInitialMemory (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (jobs : List Nat) (mem : Memory) : Memory :=
  jobs.foldl (fun previous pid => applyInitialProgram cfg plans p initial pid previous) mem

def completeInitialMemory (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) : Memory :=
  scheduledInitialMemory cfg plans p initial (List.range (gridCount p)) initial.mem

/-- Fresh program registers and source-derived pids for each actual execution;
the previous actual memory is threaded through every program. -/
def runScheduleAux (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) : List Nat → State Word → Option (State Word)
  | [], current => some current
  | pid :: jobs, current =>
      (FP.Structural.exec (engine rest cfg plans) (source p)
        (programState p initial current.mem pid)).bind
          (runScheduleAux rest cfg plans p initial jobs)

def runSchedule (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (jobs : List Nat) : Option (State Word) :=
  runScheduleAux rest cfg plans p initial jobs initial

theorem schedule_aux_correct (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (p : Parameters) (initial : State Word)
    (separate : InputsSeparate p) (jobs : List Nat) (current : State Word)
    (valid : ∀ pid ∈ jobs, pid < gridCount p)
    (ha : current.mem p.A = initial.mem p.A) (hb : current.mem p.B = initial.mem p.B) :
    ∃ final, runScheduleAux rest cfg plans p initial jobs current = some final ∧
      final.mem = scheduledInitialMemory cfg plans p initial jobs current.mem := by
  induction jobs generalizing current with
  | nil => exact ⟨current, rfl, rfl⟩
  | cons pid jobs ih =>
    obtain ⟨next, he, hm, hA, hB⟩ := complete_preserves_inputs rest cfg plans p
      (programState p initial current.mem pid) separate
    have nA : next.mem p.A = initial.mem p.A := hA.trans ha
    have nB : next.mem p.B = initial.mem p.B := hB.trans hb
    obtain ⟨final, hr, hf⟩ := ih next (fun k hk => valid k (List.mem_cons_of_mem _ hk)) nA nB
    refine ⟨final, ?_, ?_⟩
    · simp only [runScheduleAux, he, Option.bind_some]
      exact hr
    · rw [hf, hm, program_expected_memory_from_initial cfg plans p initial current.mem
        pid (valid pid (List.mem_cons_self)) ha hb]
      rfl

theorem schedule_correct (rest : Algebra Word) (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (separate : InputsSeparate p) (jobs : List Nat)
    (valid : ∀ pid ∈ jobs, pid < gridCount p) :
    ∃ final, runSchedule rest cfg plans p initial jobs = some final ∧
      final.mem = scheduledInitialMemory cfg plans p initial jobs initial.mem :=
  schedule_aux_correct rest cfg plans p initial separate jobs initial valid rfl rfl

theorem scheduled_memory_perm (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (left right : List Nat) (mem : Memory)
    (valid : ∀ pid ∈ left, pid < gridCount p) (order : left.Perm right) :
    scheduledInitialMemory cfg plans p initial left mem =
      scheduledInitialMemory cfg plans p initial right mem := by
  apply order.foldl_eq'
  intro k hk l hl previous
  exact initial_programs_commute cfg plans p initial previous k l (valid k hk) (valid l hl)

theorem scheduled_memory_frame (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (jobs : List Nat) (mem : Memory)
    (r : RegionName) (o : Nat) (outside : ∀ pid ∈ jobs, ¬ProgramWrites p initial pid r o) :
    scheduledInitialMemory cfg plans p initial jobs mem r o = mem r o := by
  classical
  induction jobs generalizing mem with
  | nil => rfl
  | cons pid jobs ih =>
    change scheduledInitialMemory cfg plans p initial jobs
      (applyInitialProgram cfg plans p initial pid mem) r o = _
    rw [ih _ (fun k hk => outside k (List.mem_cons_of_mem _ hk))]
    exact if_neg (outside pid List.mem_cons_self)

theorem grid_outside_not_written (p : Parameters) (initial : State Word)
    (pid : Nat) (valid : pid < gridCount p) (r : RegionName) (o : Nat)
    (outside : GridOutside p r o) : ¬ProgramWrites p initial pid r o := by
  have localOutside := outside_program p initial initial.mem pid valid r o outside
  rintro ⟨region, i, same⟩
  rcases localOutside with regionOut | offsetOut
  · exact regionOut region
  · exact offsetOut i same

theorem complete_schedule_correct (rest : Algebra Word) (cfg : FP.Config)
    (plans : ∀ n, ReductionPlan n) (p : Parameters) (initial : State Word)
    (separate : InputsSeparate p) (jobs : List Nat)
    (order : jobs.Perm (List.range (gridCount p))) :
    ∃ final, runSchedule rest cfg plans p initial jobs = some final ∧
      final.mem = completeInitialMemory cfg plans p initial ∧
      (∀ r o, GridOutside p r o → final.mem r o = initial.mem r o) := by
  have valid : ∀ pid ∈ jobs, pid < gridCount p := by
    intro pid hp
    exact List.mem_range.mp (order.mem_iff.mp hp)
  obtain ⟨final, he, hm⟩ := schedule_correct rest cfg plans p initial separate jobs valid
  refine ⟨final, he, hm.trans (scheduled_memory_perm cfg plans p initial jobs _ initial.mem valid order), ?_⟩
  intro r o outside
  rw [hm]
  apply scheduled_memory_frame
  intro pid hp
  exact grid_outside_not_written p initial pid (valid pid hp) r o outside

/-- The new initial-input reference agrees with the previously published
ordered current-memory grid reference under the explicit separated profile. -/
theorem reference_grid_prefix_from_initial (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (separate : InputsSeparate p)
    (k : Nat) (hk : k ≤ gridCount p) :
    referenceGridPrefix cfg plans p initial k =
      scheduledInitialMemory cfg plans p initial (List.range k) initial.mem := by
  induction k with
  | zero => rfl
  | succ k ih =>
    have ha : referenceGridPrefix cfg plans p initial k p.A = initial.mem p.A := by
      funext o
      exact grid_reference_frame cfg plans p initial k (by omega) p.A o (Or.inl separate.1)
    have hb : referenceGridPrefix cfg plans p initial k p.B = initial.mem p.B := by
      funext o
      exact grid_reference_frame cfg plans p initial k (by omega) p.B o (Or.inl separate.2)
    rw [referenceGridPrefix, program_expected_memory_from_initial cfg plans p initial
      (referenceGridPrefix cfg plans p initial k) k (by omega) ha hb, ih (by omega)]
    simp only [scheduledInitialMemory, List.range_succ, List.foldl_append, List.foldl_cons,
      List.foldl_nil]

theorem complete_initial_memory_eq_grid (cfg : FP.Config) (plans : ∀ n, ReductionPlan n)
    (p : Parameters) (initial : State Word) (separate : InputsSeparate p) :
    completeInitialMemory cfg plans p initial = gridExpectedMemory cfg plans p initial :=
  (reference_grid_prefix_from_initial cfg plans p initial separate (gridCount p) (by rfl)).symm

#specNonCircular completeInitialMemory avoiding [runSchedule, runScheduleAux, source,
  FP.Structural.exec, FP.Structural.run]
#axiomsClean program_expected_memory_from_initial
#axiomsClean initial_program_preserves_inputs
#axiomsClean initial_programs_commute
#axiomsClean schedule_aux_correct
#axiomsClean schedule_correct
#axiomsClean scheduled_memory_perm
#axiomsClean scheduled_memory_frame
#axiomsClean grid_outside_not_written
#axiomsClean complete_schedule_correct
#axiomsClean reference_grid_prefix_from_initial
#axiomsClean complete_initial_memory_eq_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BatchedVecmatMult.ValueProofs

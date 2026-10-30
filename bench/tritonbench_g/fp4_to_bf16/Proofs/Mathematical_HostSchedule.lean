import bench.tritonbench_g.fp4_to_bf16.Proofs.Mathematical_HostCorrect

/-! Order independence for the complete original logical FP4 grid. Every
scheduled program executes the existing source with fresh registers and the
preceding program's memory. The final result reads only the initial packed
input. This proves arbitrary sequential program schedules, not native GPU
instruction interleavings or a physical-memory/allocator refinement. -/
namespace VeriTile.Bench.TritonBenchG.Fp4ToBf16.HostSchedule
open VeriTile.Triton FP.Structural FPValue Host
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

abbrev Writes (h : Input) (pid : Nat) (r : RegionName) (o : Nat) : Prop :=
  r = h.output ∧ o < h.shape.count * 2 ∧ o / 1024 = pid

private theorem program_value (h : Input) (initial : State Word) (pid : Nat)
    (j : TileIndex [(parameters h).block * 2]) :
    expectedValue (parameters h) (programState h pid initial) j =
      value h initial (pid * 1024 + j.1.val) := by
  have hd : (pid * 1024 + j.1.val) / 2 = pid * 512 + j.1.val / 2 := by omega
  have hm : (pid * 1024 + j.1.val) % 2 = j.1.val % 2 := by omega
  simp only [expectedValue, initialNibble, inputOffset, parameters, programState,
    value, nibble, hd, hm, ↓reduceIte]

/-- Exact memory of one actual source execution, with no assumption about
which other programs have already executed. -/
theorem program_memory (rest : Algebra Word) (h : Input) (initial : State Word)
    (typed : PackedInput h initial) (pid : Nat) (r : RegionName) (o : Nat) :
    (execution rest (parameters h) (programState h pid initial)).val.mem r o =
      if Writes h pid r o then cell h initial o else initial.mem r o := by
  let start := programState h pid initial
  have ht : ActiveBytes (parameters h) start := by
    intro i hi
    have hi' : pid * 512 + i.val < h.shape.count := by
      simpa only [inputOffset, parameters, start, programState, ↓reduceIte] using hi
    obtain ⟨b, hb⟩ := typed (pid * 512 + i.val) hi'
    exact ⟨b, hb⟩
  rw [execution_memory rest (parameters h) start ht]
  by_cases here : Writes h pid r o
  · obtain ⟨hr, hn, hp⟩ := here
    subst r
    let j : TileIndex [(parameters h).block * 2] :=
      (⟨o % 1024, by dsimp [parameters]; omega⟩, PUnit.unit)
    have hoff : outputOffset (parameters h) start j = o := by
      dsimp [outputOffset, parameters, start, programState, j]
      omega
    have ha : active (parameters h) start j := by
      change outputOffset (parameters h) start j < h.shape.count * 2
      rw [hoff]
      exact hn
    have hv := readback (parameters h) start j ha
    rw [hoff] at hv
    change (written (parameters h) start).mem h.output o = _ at hv
    rw [hv]
    have he := program_value h initial pid j
    have hj : pid * 1024 + j.1.val = o := by dsimp [j]; omega
    rw [hj] at he
    change Cell.mk .bf16 (expectedValue (parameters h) start j) = _
    rw [he]
    simp only [Writes, hn, hp, and_self, ↓reduceIte, cell]
  · have untouched : r ≠ (parameters h).output ∨
        ∀ j : TileIndex [(parameters h).block * 2], active (parameters h) start j →
          o ≠ outputOffset (parameters h) start j := by
      by_cases hr : r = h.output
      · right
        intro j ha he
        have hj := j.1.isLt
        dsimp [active, outputOffset, parameters, start, programState] at ha he hj
        apply here
        refine ⟨hr, ?_, ?_⟩ <;> omega
      · exact Or.inl hr
    rw [frame (parameters h) start r o untouched]
    simp only [here, ↓reduceIte, start, programState]

def runSchedule (rest : Algebra Word) (h : Input) :
    List Nat → State Word → Option (State Word)
  | [], initial => some initial
  | pid :: jobs, initial =>
      (FP.Structural.exec (backend rest) (source (parameters h))
        (programState h pid initial)).bind (runSchedule rest h jobs)

/-- Exact output for any partial schedule, computed from initial input only.
Repeated program IDs are harmless; programs beyond the grid have no active
stores. Neither property is assumed by the execution proof. -/
def scheduledMemory (h : Input) (jobs : List Nat) (initial : State Word) :
    RegionName → Nat → Cell Word :=
  fun r o => if r = h.output ∧ o < h.shape.count * 2 ∧ o / 1024 ∈ jobs
    then cell h initial o else initial.mem r o

theorem schedule_correct (rest : Algebra Word) (h : Input) (jobs : List Nat)
    (initial : State Word) (fresh : h.input ≠ h.output) (typed : PackedInput h initial) :
    ∃ final, runSchedule rest h jobs initial = some final ∧
      final.mem = scheduledMemory h jobs initial := by
  induction jobs generalizing initial with
  | nil => exact ⟨initial, rfl, by funext r o; simp [scheduledMemory]⟩
  | cons pid jobs ih =>
    let next := (execution rest (parameters h) (programState h pid initial)).val
    have hn : FP.Structural.exec (backend rest) (source (parameters h))
        (programState h pid initial) = some next :=
      (execution rest (parameters h) (programState h pid initial)).property
    have same : next.mem h.input = initial.mem h.input := by
      funext o
      simpa only [Writes, fresh, false_and, ↓reduceIte] using
        program_memory rest h initial typed pid h.input o
    have ht : PackedInput h next := by
      intro i hi
      rw [same]
      exact typed i hi
    obtain ⟨final, hr, hm⟩ := ih next ht
    refine ⟨final, by simp only [runSchedule, hn, Option.bind_some]; exact hr, ?_⟩
    rw [hm]
    funext r o
    have hc : cell h next o = cell h initial o := by
      simp only [cell, value, nibble, same]
    unfold scheduledMemory
    rw [hc]
    change (if _ then _ else (execution rest (parameters h) (programState h pid initial)).val.mem r o) = _
    rw [program_memory rest h initial typed pid r o]
    simp only [Writes, List.mem_cons]
    split_ifs <;> simp_all

/-- Any schedule containing every original program produces the full exact
host reference. The actual grid is not reduced to one program or one block. -/
theorem complete_schedule (rest : Algebra Word) (h : Input) (jobs : List Nat)
    (initial : State Word) (fresh : h.input ≠ h.output) (typed : PackedInput h initial)
    (cover : ∀ pid, pid < grid h → pid ∈ jobs) :
    ∃ final, runSchedule rest h jobs initial = some final ∧
      final.mem = Host.expectedMemory h initial := by
  obtain ⟨final, hr, hm⟩ := schedule_correct rest h jobs initial fresh typed
  refine ⟨final, hr, hm.trans ?_⟩
  funext r o
  have total := grid_covers h
  by_cases active : r = h.output ∧ o < h.shape.count * 2
  · have pidBound : o / 1024 < grid h := by omega
    have listed := cover (o / 1024) pidBound
    simp only [scheduledMemory, Host.expectedMemory, active.1, active.2, listed, and_self, ↓reduceIte]
  · have missed : ¬(r = h.output ∧ o < h.shape.count * 2 ∧ o / 1024 ∈ jobs) := by tauto
    simp only [scheduledMemory, Host.expectedMemory, active, missed, ↓reduceIte]

theorem distinct_program_writes (h : Input) (p q : Nat) (different : p ≠ q)
    (r : RegionName) (o : Nat) : ¬(Writes h p r o ∧ Writes h q r o) := by
  rintro ⟨hp,hq⟩
  exact different (hp.2.2.symm.trans hq.2.2)

theorem writes_preserve_input_region (h : Input) (fresh : h.input ≠ h.output)
    (pid : Nat) (o : Nat) : ¬Writes h pid h.input o := by
  intro writes
  exact fresh writes.1

#specNonCircular scheduledMemory avoiding [runSchedule, execution, FP.Structural.exec, source]
#axiomsClean program_memory
#axiomsClean schedule_correct
#axiomsClean complete_schedule
#axiomsClean distinct_program_writes
#axiomsClean writes_preserve_input_region
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.Fp4ToBf16.HostSchedule

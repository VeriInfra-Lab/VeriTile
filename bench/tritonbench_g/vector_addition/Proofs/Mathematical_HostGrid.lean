import bench.tritonbench_g.vector_addition.Sources.Mathematical_Value

/-! The wrapper's ceil-divided grid, with every program reading memory left
by its predecessors. Distinct flat tile windows justify the final initial-
memory formula even if input/output logical region names coincide. This is
an ordered logical launch, not a GPU instruction-interleaving theorem. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs
open VeriTile.Triton FP.Structural
noncomputable section

def gridCount (p : Parameters) := (p.n + p.block - 1) / p.block

def programState {α : Type} (p : Parameters) (initial : State α)
    (mem : RegionName → Nat → Cell α) (k : Nat) : State α :=
  { initial with
    mem := mem
    regs := fun _ _ _ => none
    pids := fun ax => if ax = 0 then k else 0
    numPids := fun ax => if ax = 0 then gridCount p else 1 }

def runGridPrefix {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | k+1 => (runGridPrefix M p initial k).bind fun previous =>
      FP.Structural.exec M (source p) (programState p initial previous.mem k)

def runGrid {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (initial : State α) :=
  runGridPrefix M p initial (gridCount p)

/-- Independent whole-vector prefix formula, not a fold over executions. -/
def prefixMemory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (k : Nat) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < k*p.block ∧ o < p.n then
    resultCell M p initial.mem o else initial.mem r o

def gridExpectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.n then resultCell M p initial.mem o else initial.mem r o

/-- Earlier output stores do not touch any input address in the next tile,
including when x, y and output share a logical region. -/
theorem next_tile_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (k o : Nat) (ho : k*p.block ≤ o) :
    resultCell M p (prefixMemory M p initial k) o = resultCell M p initial.mem o := by
  simp [resultCell, prefixMemory, Nat.not_lt.mpr ho]

theorem reference_step {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (k : Nat) :
    expectedMemory M p (programState p initial (prefixMemory M p initial k) k) =
      prefixMemory M p initial (k+1) := by
  funext r o
  rw [expected_cell]
  simp only [programState, ↓reduceIte]
  by_cases before : o < k*p.block
  · have later : o < (k+1)*p.block := by rw [Nat.add_mul]; simp only [Nat.one_mul]; omega
    simp [prefixMemory, before, later, Nat.not_le.mpr before]
  · have lower : k*p.block ≤ o := Nat.le_of_not_gt before
    rw [next_tile_inputs M p initial k o lower]
    simp [prefixMemory, before, lower, Nat.add_mul]

theorem grid_prefix_correct {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (k : Nat) :
    ∃ final, runGridPrefix M p initial k = some final ∧ final.mem = prefixMemory M p initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, by funext r o; simp [prefixMemory]⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    obtain ⟨final, he, hf⟩ := complete_correct M p (programState p initial previous.mem k)
    refine ⟨final, ?_, ?_⟩
    · simp only [runGridPrefix, hr, Option.bind_some]
      exact he
    · rw [hf, hm, reference_step]

theorem grid_covers (p : Parameters) (positive : 0 < p.block) :
    p.n ≤ gridCount p * p.block := by
  have decomposition := Nat.mod_add_div (p.n+p.block-1) p.block
  have remainder := Nat.mod_lt (p.n+p.block-1) positive
  unfold gridCount
  rw [Nat.mul_comm] at decomposition
  omega

theorem grid_reference_complete {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (positive : 0 < p.block) :
    prefixMemory M p initial (gridCount p) = gridExpectedMemory M p initial := by
  have covers := grid_covers p positive
  funext r o
  by_cases ho : o < p.n
  · have hgrid : o < gridCount p * p.block := lt_of_lt_of_le ho covers
    simp [prefixMemory, gridExpectedMemory, ho, hgrid]
  · simp [prefixMemory, gridExpectedMemory, ho]

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (initial : State α) (positive : 0 < p.block) :
    ∃ final, runGrid M p initial = some final ∧ final.mem = gridExpectedMemory M p initial := by
  obtain ⟨final, he, hm⟩ := grid_prefix_correct M p initial (gridCount p)
  exact ⟨final, he, hm.trans (grid_reference_complete M p initial positive)⟩

#specNonCircular gridExpectedMemory avoiding [source, runGrid, runGridPrefix, FP.Structural.exec]
#axiomsClean next_tile_inputs
#axiomsClean reference_step
#axiomsClean grid_prefix_correct
#axiomsClean grid_covers
#axiomsClean grid_reference_complete
#axiomsClean grid_correct
end
end VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs

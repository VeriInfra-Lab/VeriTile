import bench.repairs.tritonbench_g.dropout_triton.Proofs.Typed

/-! Complete ordered grid for typed floating source materializations. Prior
stores cannot change later tile input addresses, even for equal region names.
This does not assert native GPU scheduling, physical layout or allocation. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.Typed
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition
open ValueProofs (Parameters offset gridCount programState grid_covers)
noncomputable section

def runGridPrefix {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | k+1 => (runGridPrefix M q p initial k).bind fun previous =>
      FP.Structural.exec (castBackend M) (source q p) (programState q initial previous.mem k)

def runGrid {α : Type} [Inhabited α] (M : Algebra α) (q : Parameters) (p : ℝ) (initial : State α) :=
  runGridPrefix M q p initial (ValueProofs.gridCount q)

/-- Independent whole-vector prefix formula, not a fold over executions. -/
def prefixMemory {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) (k : Nat) : RegionName → Nat → Cell α :=
  fun r o => if r = q.output ∧ o < k*q.block ∧ o < q.n then
    resultCell M q p initial.mem o else initial.mem r o

def gridExpectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) : RegionName → Nat → Cell α :=
  fun r o => if r = q.output ∧ o < q.n then resultCell M q p initial.mem o else initial.mem r o

/-- Earlier output stores do not touch any input address in the next tile,
including when x, y and output share a logical region. -/
theorem next_tile_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) (k o : Nat) (ho : k*q.block ≤ o) :
    resultCell M q p (prefixMemory M q p initial k) o = resultCell M q p initial.mem o := by
  simp [resultCell, prefixMemory, Nat.not_lt.mpr ho]

theorem reference_step {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) (k : Nat) :
    expectedMemory M q p (programState q initial (prefixMemory M q p initial k) k) =
      prefixMemory M q p initial (k+1) := by
  funext r o
  rw [expected_cell]
  simp only [programState, ↓reduceIte]
  by_cases before : o < k*q.block
  · have later : o < (k+1)*q.block := by rw [Nat.add_mul]; simp only [Nat.one_mul]; omega
    simp [prefixMemory, before, later, Nat.not_le.mpr before]
  · have lower : k*q.block ≤ o := Nat.le_of_not_gt before
    rw [next_tile_inputs M q p initial k o lower]
    simp [prefixMemory, before, lower, Nat.add_mul]

theorem grid_prefix_correct {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) (k : Nat) :
    ∃ final, runGridPrefix M q p initial k = some final ∧ final.mem = prefixMemory M q p initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, by funext r o; simp [prefixMemory]⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    obtain ⟨final, he, hf⟩ := complete_correct M q p (programState q initial previous.mem k)
    refine ⟨final, ?_, ?_⟩
    · simp only [runGridPrefix, hr, Option.bind_some]
      exact he
    · rw [hf, hm, reference_step]

theorem grid_reference_complete {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) (positive : 0 < q.block) :
    prefixMemory M q p initial (ValueProofs.gridCount q) = gridExpectedMemory M q p initial := by
  have covers := grid_covers q positive
  funext r o
  by_cases ho : o < q.n
  · have hgrid : o < ValueProofs.gridCount q * q.block := lt_of_lt_of_le ho covers
    simp [prefixMemory, gridExpectedMemory, ho, hgrid]
  · simp [prefixMemory, gridExpectedMemory, ho]

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α)
    (q : Parameters) (p : ℝ) (initial : State α) (positive : 0 < q.block) :
    ∃ final, runGrid M q p initial = some final ∧ final.mem = gridExpectedMemory M q p initial := by
  obtain ⟨final, he, hm⟩ := grid_prefix_correct M q p initial (ValueProofs.gridCount q)
  exact ⟨final, he, hm.trans (grid_reference_complete M q p initial positive)⟩

#specNonCircular gridExpectedMemory avoiding [source, runGrid, runGridPrefix, FP.Structural.exec]
#axiomsClean next_tile_inputs
#axiomsClean reference_step
#axiomsClean grid_prefix_correct
#axiomsClean grid_reference_complete
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.DropoutTriton.Typed

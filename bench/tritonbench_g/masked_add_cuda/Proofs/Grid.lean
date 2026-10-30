import bench.tritonbench_g.masked_add_cuda.Proofs.Typed

/-! Complete ordered grid for typed floating source materializations. Prior
stores cannot change later tile input addresses, even for equal region names.
This does not assert native GPU scheduling, physical layout or allocation. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.VectorAddition.ValueProofs (offset gridCount programState grid_covers)
noncomputable section

def runGridPrefix {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) : Nat → Option (State α)
  | 0 => some initial
  | k+1 => (runGridPrefix M storage p alpha initial k).bind fun previous =>
      FP.Structural.exec M (candidateKernel storage p alpha) (programState p initial previous.mem k)

def runGrid {α : Type} [Inhabited α] (M : Algebra α) (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) :=
  runGridPrefix M storage p alpha initial (gridCount p)

/-- Independent whole-vector prefix formula, not a fold over executions. -/
def prefixMemory {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) (k : Nat) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < k*p.block ∧ o < p.n ∧ (initial.mem p.y o).read .int = 0 then
    resultCell M p alpha initial.mem o else initial.mem r o

def gridExpectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.n ∧ (initial.mem p.y o).read .int = 0 then resultCell M p alpha initial.mem o else initial.mem r o

/-- Earlier output stores do not touch any input address in the next tile,
including when x, y and output share a logical region. -/
theorem next_tile_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) (k o : Nat) (ho : k*p.block ≤ o) :
    resultCell M p alpha (prefixMemory M storage p alpha initial k) o = resultCell M p alpha initial.mem o := by
  simp [resultCell, prefixMemory, Nat.not_lt.mpr ho]

theorem reference_step {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) (k : Nat) :
    expectedMemory M p alpha (programState p initial (prefixMemory M storage p alpha initial k) k) =
      prefixMemory M storage p alpha initial (k+1) := by
  funext r o
  rw [expected_cell]
  simp only [programState, ↓reduceIte]
  by_cases before : o < k*p.block
  · have later : o < (k+1)*p.block := by rw [Nat.add_mul]; simp only [Nat.one_mul]; omega
    simp [prefixMemory, before, later, Nat.not_le.mpr before]
  · have lower : k*p.block ≤ o := Nat.le_of_not_gt before
    rw [next_tile_inputs M storage p alpha initial k o lower]
    simp [prefixMemory, before, lower, Nat.add_mul]

theorem grid_prefix_correct {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) (k : Nat) :
    ∃ final, runGridPrefix M storage p alpha initial k = some final ∧ final.mem = prefixMemory M storage p alpha initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, by funext r o; simp [prefixMemory]⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    obtain ⟨final, he, hf⟩ := complete_correct M storage p alpha (programState p initial previous.mem k)
    refine ⟨final, ?_, ?_⟩
    · simp only [runGridPrefix, hr, Option.bind_some]
      exact he
    · rw [hf, hm, reference_step]

theorem grid_reference_complete {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) (positive : 0 < p.block) :
    prefixMemory M storage p alpha initial (gridCount p) = gridExpectedMemory M storage p alpha initial := by
  have covers := grid_covers p positive
  funext r o
  by_cases ho : o < p.n
  · have hgrid : o < gridCount p * p.block := lt_of_lt_of_le ho covers
    simp [prefixMemory, gridExpectedMemory, ho, hgrid]
  · simp [prefixMemory, gridExpectedMemory, ho]

theorem grid_correct {α : Type} [Inhabited α] (M : Algebra α)
    (storage : MaskStorage) (p : Parameters) (alpha : ℝ) (initial : State α) (positive : 0 < p.block) :
    ∃ final, runGrid M storage p alpha initial = some final ∧ final.mem = gridExpectedMemory M storage p alpha initial := by
  obtain ⟨final, he, hm⟩ := grid_prefix_correct M storage p alpha initial (gridCount p)
  exact ⟨final, he, hm.trans (grid_reference_complete M storage p alpha initial positive)⟩

#specNonCircular gridExpectedMemory avoiding [candidateKernel, runGrid, runGridPrefix, FP.Structural.exec]
#axiomsClean next_tile_inputs
#axiomsClean reference_step
#axiomsClean grid_prefix_correct
#axiomsClean grid_reference_complete
#axiomsClean grid_correct
end
end VeriTile.Bench.Repairs.MaskedAddCuda

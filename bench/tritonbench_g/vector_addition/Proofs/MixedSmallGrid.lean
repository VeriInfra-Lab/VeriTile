import bench.tritonbench_g.vector_addition.Proofs.MixedSmallValue

/-! Complete ordered grid for mixed integer/bool and fp16/bf16 source materializations. Prior
stores cannot change later tile input addresses, even for equal region names.
This does not assert native GPU scheduling, physical layout or allocation. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall
open VeriTile.Triton FP.Structural
open IntegerSource (Storage)
open ValueProofs (Parameters offset gridCount programState grid_covers)
noncomputable section
local instance : Inhabited Word := ⟨FP.Value.zero .fp32⟩

def runGridPrefix (rest : Algebra Word)
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) : Nat → Option (State Word)
  | 0 => some initial
  | k+1 => (runGridPrefix rest cfg f us uu integerFirst d p initial k).bind fun previous =>
      FP.Structural.exec (backend rest cfg f us uu) (source f integerFirst d p) (programState p initial previous.mem k)

def runGrid (rest : Algebra Word) (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) :=
  runGridPrefix rest cfg f us uu integerFirst d p initial (gridCount p)

/-- Independent whole-vector prefix formula, not a fold over executions. -/
def prefixMemory
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) (k : Nat) : RegionName → Nat → Cell Word :=
  fun r o => if r = p.output ∧ o < k*p.block ∧ o < p.n then
    resultCell cfg f us uu integerFirst d p initial.mem o else initial.mem r o

def gridExpectedMemory
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) : RegionName → Nat → Cell Word :=
  fun r o => if r = p.output ∧ o < p.n then resultCell cfg f us uu integerFirst d p initial.mem o else initial.mem r o

/-- Earlier output stores do not touch any input address in the next tile,
including when x, y and output share a logical region. -/
theorem next_tile_inputs
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) (k o : Nat) (ho : k*p.block ≤ o) :
    resultCell cfg f us uu integerFirst d p (prefixMemory cfg f us uu integerFirst d p initial k) o = resultCell cfg f us uu integerFirst d p initial.mem o := by
  simp [resultCell, sumValue, prefixMemory, Nat.not_lt.mpr ho]

theorem reference_step
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) (k : Nat) :
    expectedMemory cfg f us uu integerFirst d p (programState p initial (prefixMemory cfg f us uu integerFirst d p initial k) k) =
      prefixMemory cfg f us uu integerFirst d p initial (k+1) := by
  funext r o
  rw [expected_cell]
  simp only [programState, ↓reduceIte]
  by_cases before : o < k*p.block
  · have later : o < (k+1)*p.block := by rw [Nat.add_mul]; simp only [Nat.one_mul]; omega
    simp [prefixMemory, before, later, Nat.not_le.mpr before]
  · have lower : k*p.block ≤ o := Nat.le_of_not_gt before
    rw [next_tile_inputs cfg f us uu integerFirst d p initial k o lower]
    simp [prefixMemory, before, lower, Nat.add_mul]

theorem grid_prefix_correct (rest : Algebra Word)
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) (k : Nat) :
    ∃ final, runGridPrefix rest cfg f us uu integerFirst d p initial k = some final ∧ final.mem = prefixMemory cfg f us uu integerFirst d p initial k := by
  induction k with
  | zero => exact ⟨initial, rfl, by funext r o; simp [prefixMemory]⟩
  | succ k ih =>
    obtain ⟨previous, hr, hm⟩ := ih
    obtain ⟨final, he, hf⟩ := complete_correct rest cfg f us uu integerFirst d p (programState p initial previous.mem k)
    refine ⟨final, ?_, ?_⟩
    · simp only [runGridPrefix, hr, Option.bind_some]
      exact he
    · rw [hf, hm, reference_step]

theorem grid_reference_complete
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) (positive : 0 < p.block) :
    prefixMemory cfg f us uu integerFirst d p initial (gridCount p) = gridExpectedMemory cfg f us uu integerFirst d p initial := by
  have covers := grid_covers p positive
  funext r o
  by_cases ho : o < p.n
  · have hgrid : o < gridCount p * p.block := lt_of_lt_of_le ho covers
    simp [prefixMemory, gridExpectedMemory, ho, hgrid]
  · simp [prefixMemory, gridExpectedMemory, ho]

theorem grid_correct (rest : Algebra Word)
    (cfg : FP.Config) (f : FloatStorage) (us : Nat → Word → Int) (uu : Nat → Word → Nat)
    (integerFirst : Bool) (d : Storage) (p : Parameters) (initial : State Word) (positive : 0 < p.block) :
    ∃ final, runGrid rest cfg f us uu integerFirst d p initial = some final ∧ final.mem = gridExpectedMemory cfg f us uu integerFirst d p initial := by
  obtain ⟨final, he, hm⟩ := grid_prefix_correct rest cfg f us uu integerFirst d p initial (gridCount p)
  exact ⟨final, he, hm.trans (grid_reference_complete cfg f us uu integerFirst d p initial positive)⟩

#specNonCircular gridExpectedMemory avoiding [source, runGrid, runGridPrefix, FP.Structural.exec]
#axiomsClean next_tile_inputs
#axiomsClean reference_step
#axiomsClean grid_prefix_correct
#axiomsClean grid_reference_complete
#axiomsClean grid_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.MixedSmall

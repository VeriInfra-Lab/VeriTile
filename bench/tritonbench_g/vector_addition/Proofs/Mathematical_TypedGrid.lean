import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedFloat

/-! Complete ordered grid for typed floating source materializations. Prior
stores cannot change later tile input addresses, even for equal region names.
This does not assert native GPU scheduling, physical layout or allocation. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat
open VeriTile.Triton FP.Structural
open ValueProofs (Parameters offset gridCount programState grid_covers)
noncomputable section

/-- Independent whole-vector prefix formula, not a fold over executions. -/
def prefixMemory {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α) (k : Nat) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < k*p.block ∧ o < p.n then
    resultCell M dx dy dc p initial.mem o else initial.mem r o

def gridExpectedMemory {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α) : RegionName → Nat → Cell α :=
  fun r o => if r = p.output ∧ o < p.n then resultCell M dx dy dc p initial.mem o else initial.mem r o

/-- Earlier output stores do not touch any input address in the next tile,
including when x, y and output share a logical region. -/
theorem next_tile_inputs {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α) (k o : Nat) (ho : k*p.block ≤ o) :
    resultCell M dx dy dc p (prefixMemory M dx dy dc p initial k) o = resultCell M dx dy dc p initial.mem o := by
  simp [resultCell, prefixMemory, Nat.not_lt.mpr ho]

theorem reference_step {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α) (k : Nat) :
    expectedMemory M dx dy dc p (programState p initial (prefixMemory M dx dy dc p initial k) k) =
      prefixMemory M dx dy dc p initial (k+1) := by
  funext r o
  rw [expected_cell]
  simp only [programState, ↓reduceIte]
  by_cases before : o < k*p.block
  · have later : o < (k+1)*p.block := by rw [Nat.add_mul]; simp only [Nat.one_mul]; omega
    simp [prefixMemory, before, later, Nat.not_le.mpr before]
  · have lower : k*p.block ≤ o := Nat.le_of_not_gt before
    rw [next_tile_inputs M dx dy dc p initial k o lower]
    simp [prefixMemory, before, lower, Nat.add_mul]

theorem grid_reference_complete {α : Type} [Inhabited α] (M : Algebra α)
    (dx dy dc : FloatDType) (p : Parameters) (initial : State α) (positive : 0 < p.block) :
    prefixMemory M dx dy dc p initial (gridCount p) = gridExpectedMemory M dx dy dc p initial := by
  have covers := grid_covers p positive
  funext r o
  by_cases ho : o < p.n
  · have hgrid : o < gridCount p * p.block := lt_of_lt_of_le ho covers
    simp [prefixMemory, gridExpectedMemory, ho, hgrid]
  · simp [prefixMemory, gridExpectedMemory, ho]

#specNonCircular gridExpectedMemory avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#axiomsClean next_tile_inputs
#axiomsClean reference_step
#axiomsClean grid_reference_complete
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat

import bench.tritonbench_g.vector_addition.Kernels
import bench.tritonbench_g.vector_addition.Proofs.Mathematical_TypedGrid

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

#specNonCircular scheduledMemory avoiding [VeriTile.Bench.Repairs.VectorAddition.Contiguous.add_kernel, FP.Structural.exec, FP.Structural.run]
#axiomsClean program_window_iff
#axiomsClean scheduled_fresh_inputs
#axiomsClean scheduled_reference_step
#axiomsClean complete_scheduled_memory
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.VectorAddition.TypedFloat

import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FP64Programs

/-! Complete sequential program permutations for the task-local typed memory
interpreter. Outputs are separated from both inputs by explicit logical-region
premises; this does not certify a native allocator or instruction interleaving. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.FP64Memory
open FP64TypedCore
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000

def gridCount (n : Nat) : Nat := (n + 1023) / blockSize

def runGrid (b : Numerics) (p : Parameters xD yD) : List Nat → Memory → Except Fault Memory
  | [], initial => .ok initial
  | pid :: pids, initial => do
    let current ← runProgram b p pid initial
    runGrid b p pids current

def footprint (pids : List Nat) : List Nat :=
  pids.flatMap (fun pid => (List.range blockSize).map (offset pid))

theorem grid_success (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (separated : Separated p) (defined : OutputDefined b p xs ys)
    (pids : List Nat) (initial : Memory) (inputs : InputWords p xs ys initial) :
    runGrid b p pids initial = .ok (referenceIndices b p xs ys (footprint pids) initial) := by
  induction pids generalizing initial with
  | nil => simp [runGrid, footprint]
  | cons pid pids ih =>
    simp only [runGrid, program_success b p xs ys initial inputs defined pid]
    change runGrid b p pids
      (referenceIndices b p xs ys ((List.range blockSize).map (offset pid)) initial) = _
    rw [ih _ (reference_inputs b p xs ys _ initial separated inputs), reference_append]
    rfl

/-- A footprint includes index i exactly when its owning program is present;
this covers the entire block, including lanes beyond the active length. -/
theorem footprint_membership (pids : List Nat) (i : Nat) :
    i ∈ footprint pids ↔ i / blockSize ∈ pids := by
  simp only [footprint, List.mem_flatMap, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨pid, hp, lane, hl, hi⟩
    have eq : i / blockSize = pid := by
      simp only [offset, blockSize] at hi hl ⊢
      omega
    simpa only [eq] using hp
  · intro hp
    refine ⟨i / blockSize, hp, i % blockSize, Nat.mod_lt _ (by decide), ?_⟩
    simp only [offset, blockSize]
    omega

theorem active_program_in_grid (n i : Nat) (active : i < n) :
    i / blockSize < gridCount n := by
  simp only [gridCount, blockSize]
  omega

theorem complete_reference (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (pids : List Nat) (complete : pids.Perm (List.range (gridCount p.n))) (initial : Memory) :
    referenceIndices b p xs ys (footprint pids) initial = reference b p xs ys initial := by
  funext r i
  by_cases active : i < p.n
  · have covered : i ∈ footprint pids := by
      rw [footprint_membership]
      exact complete.mem_iff.mpr (List.mem_range.mpr (active_program_in_grid p.n i active))
    simp [referenceIndices, reference, active, covered]
  · simp [referenceIndices, reference, active]

theorem complete_grid_success (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (separated : Separated p) (defined : OutputDefined b p xs ys)
    (pids : List Nat) (complete : pids.Perm (List.range (gridCount p.n)))
    (initial : Memory) (inputs : InputWords p xs ys initial) :
    runGrid b p pids initial = .ok (reference b p xs ys initial) := by
  rw [grid_success b p xs ys separated defined pids initial inputs,
    complete_reference b p xs ys pids complete initial]

theorem program_permutation_independent (b : Numerics) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (separated : Separated p) (defined : OutputDefined b p xs ys)
    (left right : List Nat)
    (leftComplete : left.Perm (List.range (gridCount p.n)))
    (rightComplete : right.Perm (List.range (gridCount p.n)))
    (initial : Memory) (inputs : InputWords p xs ys initial) :
    runGrid b p left initial = runGrid b p right initial := by
  rw [complete_grid_success b p xs ys separated defined left leftComplete initial inputs,
    complete_grid_success b p xs ys separated defined right rightComplete initial inputs]

/-- An explicit interpreter failure aborts the remaining mathematical grid.
This states no native trap/poison or partial-store behavior. -/
theorem grid_error (b : Numerics) (p : Parameters xD yD) (pid : Nat) (pids : List Nat)
    (initial : Memory) (fault : Fault) (failed : runProgram b p pid initial = .error fault) :
    runGrid b p (pid :: pids) initial = .error fault := by
  rw [runGrid, failed]
  rfl

theorem program_outside_grid (b : Numerics) (p : Parameters xD yD) (pid : Nat)
    (initial : Memory) (outside : gridCount p.n ≤ pid) : runProgram b p pid initial = .ok initial := by
  apply lanes_all_masked
  intro lane _
  simp only [offset, gridCount, blockSize] at outside ⊢
  omega

theorem zero_grid (b : Numerics) (p : Parameters xD yD) (initial : Memory)
    (empty : p.n = 0) : runGrid b p (List.range (gridCount p.n)) initial = .ok initial := by
  simp [gridCount, blockSize, empty, runGrid]

#specNonCircular reference avoiding [runGrid, runProgram, runLanes, laneOutcome,
  source, Expr.eval, runLane]
#specNonCircular OutputDefined avoiding [runGrid, runProgram, runLanes, laneOutcome,
  source, Expr.eval, runLane]
#specNonCircular InputWords avoiding [runGrid, runProgram, runLanes, laneOutcome,
  source, Expr.eval, runLane]
#axiomsClean grid_success
#axiomsClean footprint_membership
#axiomsClean active_program_in_grid
#axiomsClean complete_reference
#axiomsClean complete_grid_success
#axiomsClean program_permutation_independent
#axiomsClean grid_error
#axiomsClean program_outside_grid
#axiomsClean zero_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.FP64Memory

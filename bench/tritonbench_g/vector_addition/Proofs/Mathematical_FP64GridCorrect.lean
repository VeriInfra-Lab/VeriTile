import bench.tritonbench_g.vector_addition.Proofs.Mathematical_FP64Grid

/-! Public logical-memory contracts for all29 successful fp64 route profiles.
The numerical callbacks are explicit and unverified native correspondences.
Output/input region separation is a selected-domain premise, not a proved
wrapper allocation property. Complete program permutations are sequential
program schedules; native memory/compiler/instruction interleavings stay open. -/
namespace VeriTile.Bench.TritonBenchG.VectorAddition.FP64GridCorrect
open FP64TypedCore FP64Memory

specification vector_addition_fp64_defined_grid_correctness
    (b : Numerics) (xD yD : Storage) (_supported : Supported xD yD)
    (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (initial : Memory) (inputs : InputWords p xs ys initial)
    (separated : Separated p) (defined : OutputDefined b p xs ys)
    (pids : List Nat) (complete : pids.Perm (List.range (gridCount p.n))) :
    ∃ final, runGrid b p pids initial = .ok final ∧
      final = reference b p xs ys initial ∧
      (∀ r i, r ≠ p.output ∨ p.n ≤ i → final r i = initial r i) := by
  refine ⟨reference b p xs ys initial,
    complete_grid_success b p xs ys separated defined pids complete initial inputs, rfl, ?_⟩
  exact fun r i outside => reference_frame b p xs ys initial r i outside

/-- All20 floating-output profiles need no conversion domain or finiteness
premise. Arbitrary input/output bit patterns retain the chosen callback policy. -/
specification vector_addition_fp64_floating_grid_correctness
    (b : Numerics) (xD yD : Storage) (supported : Supported xD yD)
    (floating : xD.isFloating = true) (p : Parameters xD yD)
    (xs : Nat → Word (.storage xD)) (ys : Nat → Word (.storage yD))
    (initial : Memory) (inputs : InputWords p xs ys initial)
    (separated : Separated p)
    (pids : List Nat) (complete : pids.Perm (List.range (gridCount p.n))) :
    ∃ final, runGrid b p pids initial = .ok final ∧
      final = reference b p xs ys initial ∧
      (∀ r i, r ≠ p.output ∨ p.n ≤ i → final r i = initial r i) :=
  vector_addition_fp64_defined_grid_correctness b xD yD supported p xs ys initial inputs
    separated (floating_defined b p xs ys floating) pids complete

#axiomsClean vector_addition_fp64_defined_grid_correctness
#axiomsClean vector_addition_fp64_floating_grid_correctness
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.VectorAddition.FP64GridCorrect

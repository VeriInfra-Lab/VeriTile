import bench.repairs.tritonbench_g.sin_computation.Proofs.Contract
import bench.repairs.tritonbench_g.sin_computation.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.SinComputation.Contiguous.Real
open VeriTile.Triton
noncomputable section

/-- Install the mathematical input while retaining arbitrary supplied output cells. -/
def allocated (xs : Nat → ℝ) (scratch : BlockState) : BlockState :=
  {scratch with mem := fun r i => if r = "x" then .real (xs i) else scratch.mem r i}

/-- The formula successful-runtime host interpretation. -/
def runWrapper (x out : TensorMetadata) (xs : Nat → ℝ) (scratch : BlockState) :
    Except InputError (Option BlockState) := do
  checkInputs x out
  let initial := allocated xs scratch
  if x.numel = 0 then return some initial
  return RealGrid.runPrograms "x" "out" x.numel (List.range (gridCount x.numel)) initial

theorem wrapper_correct (x out : TensorMetadata) (h : InputContract x out)
    (xs : Nat → ℝ) (scratch : BlockState) :
    ∃ final, runWrapper x out xs scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.readMem "out" i = formula xs i) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i →
        final.mem r i = (allocated xs scratch).mem r i) := by
  have accepted : checkInputs x out = .ok () := (checkTensorInputs_iff x out).mpr h
  by_cases empty : x.numel = 0
  · exact ⟨allocated xs scratch, by simp [runWrapper, accepted, empty],
      by simp [empty], by intro r i _; rfl⟩
  · obtain ⟨final, he, hv, hf⟩ := RealGrid.full_grid_correct "x" "out"
      x.numel (allocated xs scratch)
    refine ⟨final, ?_, ?_, hf⟩
    · simpa [runWrapper, accepted, empty, gridCount, gridCount] using he
    · intro i hi
      simpa [allocated, BlockState.readMem, formula] using hv i hi

#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinComputation.Contiguous.Real

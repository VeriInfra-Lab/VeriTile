import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.Contract
import bench.repairs.tritonbench_g.vector_addition_custom.Proofs.RealGrid

/-! Mathematical interpretation under ordinary successful Tensor allocation and
launch semantics. `xs` and `ys` denote the original logical input values. Since
the admitted tensors are contiguous without a negative bit, these also denote
the values read at successive data-pointer offsets; no copies are modeled. -/
namespace VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.VectorAdditionCustom
noncomputable section

def allocated (xs ys : Nat → ℝ) (scratch : BlockState) : BlockState :=
  {scratch with mem := fun r i =>
    if r = "x" then .real (xs i)
    else if r = "y" then .real (ys i)
    else scratch.mem r i}

/-- Guarded host composition. Allocation/launch failure is outside this
successful-runtime model; rejected inputs do not reach the kernel. -/
def runWrapper (x y : TensorMetadata) (xs ys : Nat → ℝ) (scratch : BlockState) :
    Except InputError (Option BlockState) := do
  checkInputs x y
  let initial := allocated xs ys scratch
  if x.numel = 0 then return some initial
  return RealGrid.runPrograms "x" "y" "out" x.numel
    (List.range (gridCount x.numel)) initial

theorem wrapper_correct (x y : TensorMetadata) (h : InputContract x y)
    (xs ys : Nat → ℝ) (scratch : BlockState) :
    ∃ final, runWrapper x y xs ys scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.readMem "out" i = formula xs ys i) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i →
        final.mem r i = (allocated xs ys scratch).mem r i) := by
  have accepted := (check_tensor_arguments_iff x y).mpr h
  by_cases empty : x.numel = 0
  · exact ⟨allocated xs ys scratch, by simp [runWrapper, accepted, empty],
      by simp [empty], by intro r i _; rfl⟩
  · obtain ⟨final, he, hv, hf⟩ := RealGrid.full_grid_correct "x" "y" "out"
      x.numel (allocated xs ys scratch)
    refine ⟨final, ?_, ?_, hf⟩
    · simpa [runWrapper, accepted, empty, gridCount, RealGrid.count] using he
    · intro i hi
      simpa [allocated, BlockState.readMem, formula] using hv i hi

#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.VectorAdditionCustom.Contiguous.Real

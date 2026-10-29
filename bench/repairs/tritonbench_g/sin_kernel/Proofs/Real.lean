import bench.repairs.tritonbench_g.sin_kernel.Proofs.Contract
import bench.repairs.tritonbench_g.sin_kernel.Proofs.RealGrid

/-! Mathematical interpretation under ordinary successful Tensor allocation and
launch semantics. `xs` denotes the original logical input values. Since
the admitted tensors are contiguous without a negative bit, these also denote
the values read at successive data-pointer offsets; no copies are modeled. -/
namespace VeriTile.Bench.Repairs.SinKernel.Math
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.SinKernel
noncomputable section

def allocated (xs : Nat → ℝ) (scratch : BlockState) : BlockState :=
  {scratch with mem := fun r i =>
    if r = "x_ptr" then .real (xs i)
    else scratch.mem r i}

/-- Guarded host composition. Allocation/launch failure is outside this
successful-runtime model; rejected inputs do not reach the kernel. -/
def runWrapper (x : TensorMetadata) (xs : Nat → ℝ) (scratch : BlockState) :
    Except InputError (Option BlockState) := do
  checkInputs x
  let initial := allocated xs scratch
  if x.numel = 0 then return some initial
  return RealGrid.runPrograms "x_ptr" "output_ptr" x.numel
    (List.range (gridCount x.numel)) initial

theorem wrapper_correct (x : TensorMetadata) (h : InputContract x)
    (xs : Nat → ℝ) (scratch : BlockState) :
    ∃ final, runWrapper x xs scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.readMem "output_ptr" i = formula xs i) ∧
      (∀ (r : RegionName) i, r ≠ "output_ptr" ∨ x.numel ≤ i →
        final.mem r i = (allocated xs scratch).mem r i) := by
  have accepted : checkInputs x = .ok () := (checkTensorInputs_iff x).mpr h
  by_cases empty : x.numel = 0
  · exact ⟨allocated xs scratch, by simp [runWrapper, accepted, empty],
      by simp [empty], by intro r i _; rfl⟩
  · obtain ⟨final, he, hv, hf⟩ := RealGrid.full_grid_correct "x_ptr" "output_ptr"
      x.numel (allocated xs scratch)
    refine ⟨final, ?_, ?_, hf⟩
    · simpa [runWrapper, accepted, empty, gridCount, RealGrid.count] using he
    · intro i hi
      simpa [allocated, BlockState.readMem, formula] using hv i hi

#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SinKernel.Math

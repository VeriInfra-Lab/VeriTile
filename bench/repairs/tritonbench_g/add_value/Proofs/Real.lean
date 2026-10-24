import bench.repairs.tritonbench_g.add_value.Proofs.Contract
import bench.repairs.tritonbench_g.add_value.Proofs.Projection
import bench.tritonbench_g.add_value.Proofs.Real
import bench.tritonbench_g.add_value.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.AddValue.Contiguous.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AddValue
noncomputable section

/-- empty_like leaves arbitrary output cells; only the input is installed. -/
def allocated (xs : Nat → ℝ) (scratch : BlockState) : BlockState :=
  {scratch with mem := fun r i => if r = "x" then .real (xs i) else scratch.mem r i}

/-- Whole-grid execution of the same shared source used by both public IOs. -/
def runPrograms (n : Nat) : List Nat → BlockState → Option BlockState
  | [], s => some s
  | pid :: rest, s => do
      let t ← exec (originalKernel n).toAlgKernel (RealGrid.programState s n pid)
      runPrograms n rest t

theorem runPrograms_eq (n : Nat) (programs : List Nat) (s : BlockState) :
    runPrograms n programs s = RealGrid.runPrograms "x" "out" n programs s := by
  induction programs generalizing s with
  | nil => rfl
  | cons pid rest ih =>
    simp only [runPrograms, RealGrid.runPrograms]
    congr 1
    funext t
    exact ih t

/-- The mathematical successful-runtime host interpretation. -/
def runWrapper (x : TensorMetadata) (xs : Nat → ℝ) (scratch : BlockState) :
    Except InputError (Option BlockState) := do
  checkInput x
  let initial := allocated xs scratch
  if x.numel = 0 then return some initial
  return runPrograms x.numel (List.range (gridCount x.numel)) initial

theorem wrapper_correct (x : TensorMetadata) (h : InputContract x)
    (xs : Nat → ℝ) (scratch : BlockState) :
    ∃ final, runWrapper x xs scratch = .ok (some final) ∧
      (∀ i, i < x.numel → final.readMem "out" i = mathematical xs i) ∧
      (∀ (r : RegionName) i, r ≠ "out" ∨ x.numel ≤ i →
        final.mem r i = (allocated xs scratch).mem r i) := by
  have accepted : checkInput x = .ok () := (checkTensorInput_iff x).mpr h
  by_cases empty : x.numel = 0
  · exact ⟨allocated xs scratch, by simp [runWrapper, accepted, empty],
      by simp [empty], by intro r i _; rfl⟩
  · obtain ⟨final, he, hv, hf⟩ := RealGrid.full_grid_correct "x" "out"
      x.numel (allocated xs scratch)
    refine ⟨final, ?_, ?_, hf⟩
    · simpa [runWrapper, accepted, empty, runPrograms_eq, gridCount, RealGrid.count] using he
    · intro i hi
      simpa [allocated, BlockState.readMem, mathematical] using hv i hi

#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.AddValue.Contiguous.Real

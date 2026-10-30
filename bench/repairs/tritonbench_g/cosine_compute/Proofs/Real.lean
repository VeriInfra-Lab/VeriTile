import bench.repairs.tritonbench_g.cosine_compute.Proofs.Contract
import bench.tritonbench_g.cosine_compute.Proofs.RealGrid

namespace VeriTile.Bench.Repairs.CosineCompute.Real
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.CosineCompute
noncomputable section

/-- Allocation preserves the input and supplies arbitrary fresh output cells.
No initialization value or cosine result is assumed for empty_like. -/
def allocated (input : Nat → ℝ) (scratch : BlockState) : BlockState :=
  {scratch with mem := fun r i => if r="a" then .real (input i) else scratch.mem r i}

def runWrapper (x : TensorMetadata) (input : Nat → ℝ) (scratch : BlockState) :
    Except InputError (Option BlockState) := do
  checkTensorInput x
  let initial := allocated input scratch
  if x.numel=0 then return some initial
  return RealGrid.runPrograms "a" "b" x.numel 256 (List.range (gridCount x.numel)) initial

theorem wrapper_correct (x : TensorMetadata) (h : InputContract x)
    (input : Nat → ℝ) (scratch : BlockState) :
    ∃ final, runWrapper x input scratch = .ok (some final) ∧
      mathematical x.numel input (final.readMem "b") ∧
      (∀ (r : RegionName) i, r ≠ "b" ∨ x.numel ≤ i → final.mem r i=(allocated input scratch).mem r i) := by
  have accepted := (checkTensorInput_iff x).mpr h
  by_cases empty : x.numel=0
  · exact ⟨allocated input scratch,by simp [runWrapper,accepted,empty],
      by simp [mathematical,empty],by intros; rfl⟩
  obtain ⟨final,he,hv,hf⟩ := RealGrid.full_grid_correct "a" "b" x.numel 256
    (by decide) (allocated input scratch)
  have grid_eq : RealGrid.count x.numel 256 = gridCount x.numel := by
    simp [RealGrid.count,gridCount]
  rw [grid_eq] at he
  refine ⟨final,by simp [runWrapper,accepted,empty,he],?_,hf⟩
  intro i hi
  simpa [allocated,BlockState.readMem,MemCell.readAs,MemCell.real,MemCell.of] using hv i hi

#axiomsClean wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.CosineCompute.Real

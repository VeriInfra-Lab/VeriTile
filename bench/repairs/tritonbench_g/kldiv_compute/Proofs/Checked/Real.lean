import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Checked.Source
import bench.repairs.tritonbench_g.kldiv_compute.Proofs.Real

namespace VeriTile.Bench.Repairs.KldivCompute.Contiguous.RealProof
open VeriTile.Triton FP.Structural
noncomputable section

def allocated (xs ys : Nat → ℝ) (scratch : State ℝ) : State ℝ :=
  {scratch with mem := fun r i => if r="x" then .mk .real (xs i)
    else if r="y" then .mk .real (ys i) else scratch.mem r i}

def parameters (n : Nat) : Parameters := ⟨"x","y","out",n,1024⟩

def runWrapper (x y : TensorMetadata) (xs ys : Nat → ℝ) (scratch : State ℝ) :
    Except InputError (Option (State ℝ)) := do
  checkTensorInputs x y
  let initial := allocated xs ys scratch
  if x.numel=0 then return some initial
  return Program.grid Math.algebra (parameters x.numel) initial

theorem wrapper_correct (x y : TensorMetadata) (h : InputContract x y)
    (xs ys : Nat → ℝ) (scratch : State ℝ) :
    ∃ final,runWrapper x y xs ys scratch=.ok (some final) ∧
      (∀ i,i<x.numel → final.mem "out" i=.mk .real (mathematical (xs i) (ys i))) ∧
      (∀ r i,r≠"out" ∨ x.numel≤ i → final.mem r i=(allocated xs ys scratch).mem r i) := by
  have admitted := (checkTensorInputs_iff x y).mpr h
  by_cases empty : x.numel=0
  · exact ⟨allocated xs ys scratch,by simp [runWrapper,admitted,empty],by simp [empty],fun _ _ _ => rfl⟩
  · obtain ⟨final,execution,memory⟩ := Program.grid_complete Math.algebra (parameters x.numel)
      (allocated xs ys scratch) (by change 0<(1024:Nat); decide)
    refine ⟨final,by simp [runWrapper,admitted,empty,execution],?_,?_⟩
    · intro i hi
      rw [memory]
      simp [VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory,parameters,hi,
        Program.resultCell,Program.scalar,Math.algebra,mathematical,allocated,Cell.read]
    · intro r i outside
      rw [memory]
      simp only [VeriTile.Bench.TritonBenchG.VectorAddition.PointwisePrograms.finalMemory,parameters]
      rw [if_neg (by rcases outside with other|after <;> simp_all)]

#axiomsClean wrapper_correct
#specNonCircular mathematical avoiding [Program.grid,Source.runProgram,runWrapper]
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.KldivCompute.Contiguous.RealProof

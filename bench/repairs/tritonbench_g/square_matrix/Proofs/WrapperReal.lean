import bench.repairs.tritonbench_g.square_matrix.Proofs.Real
import bench.repairs.tritonbench_g.square_matrix.Proofs.Contract

namespace VeriTile.Bench.Repairs.SquareMatrix.HistoricalReal
open VeriTile.Triton
noncomputable section

/-- Independent mathematical operation: pointwise square, not matrix product. -/
def reference (x : ℝ) : ℝ := x*x

/-- Successful allocation metadata and its logical input binding. The initial
output contents are unconstrained because Python uses empty_like. -/
structure Allocation (x : TensorMetadata) (xs : Nat → ℝ) where
  output : TensorMetadata
  memory : BlockState
  shape : output.shape=x.shape
  device : output.device=x.device
  dtype : output.dtype=.floating .fp32
  input : ∀ i,i<x.numel → memory.readMem "x" i=xs i

def runWrapper (x : TensorMetadata) (xs : Nat → ℝ) (a : Allocation x xs) :
    Except InputError (Option (TensorMetadata × BlockState)) := do
  checkInput (.tensor x)
  if x.numel=0 then return some (a.output,a.memory)
  return (RealProof.runRows "x" "out" (cols x) (cols x) (cols x) (rows x)
    (List.range (rows x)) a.memory).map (a.output,·)

theorem python_checks_iff_preconditions (x : ArgumentMetadata) :
    checkInput x=.ok () ↔ AdmittedArgument x := checkInput_iff x

/-- Complete guarded host/grid correctness in the Real projection. Both empty
matrix shapes and every nonempty row/column are covered, with exterior frame. -/
theorem square_wrapper_correct (x : TensorMetadata) (h : InputContract x)
    (xs : Nat → ℝ) (a : Allocation x xs) :
    ∃ final, runWrapper x xs a=.ok (some (a.output,final)) ∧
      a.output.shape=x.shape ∧ a.output.device=x.device ∧ a.output.dtype=.floating .fp32 ∧
      (∀ r c,r<rows x → c<cols x → final.readMem "out" (r*cols x+c)=reference (xs (r*cols x+c))) ∧
      (∀ (region : RegionName) i,region ≠ "out" ∨ x.numel ≤ i → final.mem region i=a.memory.mem region i) := by
  have accepted : checkInput (.tensor x)=.ok () := (checkTensorInput_iff x).mpr h
  by_cases empty : x.numel=0
  · refine ⟨a.memory,by simp [runWrapper,accepted,empty],a.shape,a.device,a.dtype,?_,by intro r i _; rfl⟩
    intro r c hr hc
    have hi := flat_lt _ _ _ _ hr hc
    rw [←numel_eq x h,empty] at hi
    omega
  · obtain ⟨final,he,hv,hf⟩ := RealProof.logical_grid (cols x) (rows x) a.memory
    refine ⟨final,by simp [runWrapper,accepted,empty,he],a.shape,a.device,a.dtype,?_,?_⟩
    · intro r c hr hc
      have hi : r*cols x+c<x.numel := by rw [numel_eq x h]; exact flat_lt _ _ _ _ hr hc
      rw [hv r c hr hc,a.input _ hi]
      rfl
    · intro r i outside
      apply hf r i
      simpa only [←numel_eq x h] using outside

#specNonCircular reference avoiding [sourceKernel,runWrapper,RealProof.runRows]
#axiomsClean python_checks_iff_preconditions
#axiomsClean square_wrapper_correct
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.SquareMatrix.HistoricalReal

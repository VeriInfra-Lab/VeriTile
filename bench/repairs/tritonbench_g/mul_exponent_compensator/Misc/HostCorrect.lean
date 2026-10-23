import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Host
import bench.repairs.tritonbench_g.mul_exponent_compensator.Proofs.Grid

namespace VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Correct
open VeriTile.Triton FP.Structural Host
open VeriTile.Torch.FP32 (Tensor TensorLaws)
open TritonMul2 (Memory Operations)
open TritonMul2.Host (logical)
noncomputable section

def mathematical (input output : Nat → ℝ) (n : Nat) : Prop :=
  ∀ i,i<n → output i=input i*(2:ℝ)^112

def realOperations : Operations ℝ := ⟨fun x => x*(2:ℝ)^112⟩
def realAlgebra (rest : Algebra ℝ) : Algebra ℝ :=
  {rest with
    literal := fun _ _ x => x
    fp32Load := id
    cast := fun _ _ _ x => x
    binary :=fun precision dtype op x y => match op with
      | .mul => x*y
      | _ => rest.binary precision dtype op x y}

theorem real_source_scalar (rest : Algebra ℝ) (x : ℝ) :
    scalar (realAlgebra rest) x=x*(2:ℝ)^112 := rfl

specification python_checks_iff_preconditions (x : ArgumentMetadata) (current : Nat) (block : BlockArgument) :
    checkInputs x current block=.ok () ↔ AdmittedArguments x current block := checkInputs_iff x current block

/-- Mathematical host interpretation follows the complete widened finite
source and the checked current-device allocation; fresh output covers all n. -/
specification wrapper_real (x : Tensor) (lx : TensorLaws x) (current B : Nat)
    (h : InputContract x.metadata current B) (before : Memory ℝ) (p : Prepared x lx current B h before)
    (order : List Nat) (schedule : order.Perm (List.range (grid x.metadata.numel B))) :
    ∃ final,
      runWrapper realOperations p order=.ok (some (p.output,final)) ∧
      mathematical (logical before x) (logical final p.output) x.metadata.numel ∧
      (∀ cell,(¬∃ i,i<x.metadata.numel ∧ cell=(layout p).output+i) → final cell=before cell) :=
  wrapper_correct realOperations x lx current B h before p order schedule

/-- Complete typed AST grid, independently specialized to mathematical Real
arithmetic. Allocation scratch has no required contents. -/
specification typed_real_grid (rest : Algebra ℝ) (p : Parameters) (initial : State ℝ) (positive : 0<p.block) :
    ∃ final,runGrid (realAlgebra rest) p initial=some final ∧
      (∀ r i,final.mem r i=if r=p.output ∧ i<p.n then
        Cell.mk .real (((initial.mem p.x i).read .real)*(2:ℝ)^112) else initial.mem r i) := by
  obtain ⟨final,he,hm⟩ := grid_correct (realAlgebra rest) p initial positive
  refine ⟨final,he,?_⟩
  intro r i
  rw [hm]
  rfl

#specNonCircular mathematical avoiding [runWrapper,Native.sourceLaunch,TritonMul2.launch,runGrid]
#axiomsClean real_source_scalar
#axiomsClean wrapper_real
#axiomsClean typed_real_grid
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.MulExponentCompensator.Contiguous.Correct

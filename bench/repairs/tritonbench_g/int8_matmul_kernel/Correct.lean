import bench.repairs.tritonbench_g.int8_matmul_kernel.Kernels
import bench.repairs.tritonbench_g.int8_matmul_kernel.Proofs.Contract
import bench.tritonbench_g.int8_matmul_kernel.Whole001.Correct

namespace VeriTile.Bench.Repairs.Int8MatmulKernel.Correct
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section

def runChecked (r : Request) (aPointer bPointer cPointer : Nat) (caller : Set Nat) (before : Memory Word)
    (allocation : Allocation r caller before) (trials : List (Fin 15)) (selected : Fin 15) := do
  checkInputs r aPointer bPointer before
  if !outputChecks r cPointer then throw InputError.outputAddress
  return (allocation.returned, allocation.math.finalMemory integerArithmetic trials selected)

/-- Independent unbounded mathematical interpretation of the entire checked
wrapper. Native signed32 arithmetic is specified separately in FPEquiv. -/
specification checked_mathematical (r : Request) (aPointer bPointer cPointer : Nat) (caller : Set Nat) (before : Memory Word)
    (allocation : Allocation r caller before) (trials : List (Fin 15)) (selected : Fin 15)
    (admitted : InputContract r aPointer bPointer before) (coveredA : r.a.covered) (coveredB : r.b.covered)
    (ordinary : OrdinaryPointers r allocation.output aPointer bPointer)
    (output : outputChecks r cPointer=true)
    (selectedLegal : r.shape.legal selected) (trialsLegal : ∀ c ∈ trials, r.shape.legal c) :
    runChecked r aPointer bPointer cPointer caller before allocation trials selected =
      .ok (allocation.returned, allocation.math.finalMemory integerArithmetic trials selected) ∧
    ReturnContract r caller allocation.returned ∧
    (∀ row col, row < r.shape.m → col < r.shape.n →
      (allocation.math.finalMemory integerArithmetic trials selected
        (outAddress r.shape allocation.buffers row col)).2 = mathematicalFormula r allocation.output before row col) ∧
    (∀ p, p.1.1 ∈ caller →
      allocation.math.finalMemory integerArithmetic trials selected p = mathLift before p) := by
  have domain := whole_domain r aPointer bPointer before admitted allocation.output trials selected
    coveredA coveredB (ordinary.checked before admitted cPointer output) selectedLegal trialsLegal
  exact ⟨by simp [runChecked, (checkInputs_iff r aPointer bPointer before).2 admitted,output],
    (whole_mathematical r caller before allocation trials selected domain).2⟩

#axiomsClean checked_mathematical
end
end VeriTile.Bench.Repairs.Int8MatmulKernel.Correct

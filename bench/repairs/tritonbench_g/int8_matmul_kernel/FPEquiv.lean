import bench.repairs.tritonbench_g.int8_matmul_kernel.Kernels
import bench.repairs.tritonbench_g.int8_matmul_kernel.Proofs.Contract
import bench.tritonbench_g.int8_matmul_kernel.Whole001.FPEquiv

namespace VeriTile.Bench.Repairs.Int8MatmulKernel.FPEquiv
open VeriTile.Bench.TritonBenchG.Int8MatmulKernel.Whole001
noncomputable section

def runChecked (r : Request) (aPointer bPointer cPointer : Nat) (caller : Set Nat) (before : Memory Word)
    (allocation : Allocation r caller before) (trials : List (Fin 15)) (selected : Fin 15)
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word) := do
  checkInputs r aPointer bPointer before
  if !outputChecks r cPointer then throw InputError.outputAddress
  return (allocation.returned,
    allocation.finalMemory (nativeArithmetic cast instruction) trials selected)

/-- All cells, completed autotune trials, exact returned int32 tensor and
caller frame. The two local numerical leaves retain defined casts and each
selected instruction's overflow behavior; no no-overflow assumption is added. -/
specification checked_finite (r : Request) (aPointer bPointer cPointer : Nat) (caller : Set Nat) (before : Memory Word)
    (allocation : Allocation r caller before) (trials : List (Fin 15)) (selected : Fin 15)
    (admitted : InputContract r aPointer bPointer before) (coveredA : r.a.covered) (coveredB : r.b.covered)
    (ordinary : OrdinaryPointers r allocation.output aPointer bPointer)
    (output : outputChecks r cPointer=true)
    (selectedLegal : r.shape.legal selected) (trialsLegal : ∀ c ∈ trials, r.shape.legal c)
    (cast : Word → Byte)
    (instruction : Instruction → (Fin 32 → Byte) → (Fin 32 → Byte) → Word → Word)
    (leaves : NumericLeaves cast instruction) :
    runChecked r aPointer bPointer cPointer caller before allocation trials selected cast instruction =
      .ok (allocation.returned, allocation.finalMemory (nativeArithmetic cast instruction) trials selected) ∧
    ReturnContract r caller allocation.returned ∧
    (∀ row col, row < r.shape.m → col < r.shape.n →
      allocation.finalMemory (nativeArithmetic cast instruction) trials selected
        (outAddress r.shape allocation.buffers row col) = finiteFormula r allocation.output selected before row col) ∧
    (∀ p, p.1.1 ∈ caller →
      allocation.finalMemory (nativeArithmetic cast instruction) trials selected p = before p) := by
  have domain := whole_domain r aPointer bPointer before admitted allocation.output trials selected
    coveredA coveredB (ordinary.checked before admitted cPointer output) selectedLegal trialsLegal
  exact ⟨by simp [runChecked, (checkInputs_iff r aPointer bPointer before).2 admitted,output],
    (whole_finite r caller before allocation trials selected domain cast instruction leaves).2⟩

#axiomsClean checked_finite
end
end VeriTile.Bench.Repairs.Int8MatmulKernel.FPEquiv

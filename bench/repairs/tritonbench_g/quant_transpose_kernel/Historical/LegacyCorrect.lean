import bench.repairs.tritonbench_g.quant_transpose_kernel.Proofs.Contract
import bench.tritonbench_g.quant_transpose_kernel.Correct

namespace VeriTile.Bench.Repairs.QuantTransposeKernel.Correct
open QuantTransposeFinite (Allocations Input Config Environment)
noncomputable section

def runChecked (p : Metadata) (env : Environment) (alloc : Allocations)
    (m : QuantizeGlobalReal.Memory) (x : String) (trials : List Config) (last : Config) :
    Except InputError QuantTransposeReal.Outcome := do
  checkInputs p
  return QuantTransposeReal.executeHost env p.input alloc m x trials last

/-- Complete guarded mathematical wrapper: independent source-ordered setup,
returned transposed cells/absmax metadata and frame, or an explicit undefined
scalar observation. Zero reciprocal is not silently equated to a Real value. -/
specification checked_mathematical (p : Metadata) (env : Environment)
    (alloc : Allocations) (m : QuantizeGlobalReal.Memory) (x : String)
    (admitted : InputContract p) (metadata : QuantTransposeReal.Metadata p.input)
    (fresh : alloc.Fresh x) (trials : List Config) (last : Config) :
    runChecked p env alloc m x trials last =
      .ok (QuantTransposeReal.executeHost env p.input alloc m x trials last) ∧
    QuantTransposeReal.Metadata p.input ∧
    QuantTransposeReal.HostSpec env p.input alloc m x trials last
      (QuantTransposeReal.executeHost env p.input alloc m x trials last) := by
  exact ⟨by simp [runChecked, (checkInputs_iff p).2 admitted],
    QuantTransposeReal.quantize_global_transpose_real_source_outcomes
      env p.input alloc m x metadata fresh trials last⟩

#axiomsClean checked_mathematical
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantTransposeKernel.Correct

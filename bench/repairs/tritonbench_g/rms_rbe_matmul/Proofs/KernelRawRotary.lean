import bench.repairs.tritonbench_g.rms_rbe_matmul.Kernels
import bench.repairs.tritonbench_g.rbe_triton_transform.Correct
import bench.repairs.tritonbench_g.rbe_triton_transform.FPEquiv

namespace VeriTile.Bench.Repairs.RmsRbeMatmul.KernelRawRotary
noncomputable section

/-- Existing source-only alias to the sibling's historical bridge quotation.
Its DIM128 frequency period is separate from the main wrapper's headDim. -/
theorem source_alias : @TypedSource.rawRotary = @RbeTritonTransform.kernel := rfl

/-- Current readable source quotation used by the sibling's public execution
theorems. The actual in-file Python helper is checked separately against the
same selected source operations by source and offline lowering evidence. -/
abbrev selectedHelper (r : RbeTritonTransform.Request) (input output : VeriTile.Triton.RegionName) :=
  RbeTritonTransform.Selected.source (RbeTritonTransform.parameters r input output)

/-- Reuse of the sibling's current actual-source whole-Tensor mathematical
result. The retained helper is not called by the single-projection wrapper. -/
abbrev real := @RbeTritonTransform.Correct.kernel_correct

/-- Reuse of the sibling's current actual-source native output-byte and full
frame result, under its explicit original helper admission/launch boundary. -/
abbrev native := @RbeTritonTransform.FPEquiv.kernel_native_equivalence

#axiomsClean source_alias
#axiomsClean real
#axiomsClean native
end
end VeriTile.Bench.Repairs.RmsRbeMatmul.KernelRawRotary

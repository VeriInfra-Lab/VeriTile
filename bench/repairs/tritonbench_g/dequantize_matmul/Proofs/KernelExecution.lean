import bench.repairs.tritonbench_g.dequantize_matmul.Kernels
import VeriTile.Triton.Float.Structural
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.DequantizeMatmul.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

abbrev exec {α : Type} [Inhabited α] (M : Algebra α)
    (p : Parameters) (version : Bool) (s : State α) : Option (State α) :=
  FP.Structural.run M (kernel version p).surfaceBody s

/-- Complete source execution for arbitrary scalar floating primitives. No
success or expected-output premise is needed for this straight-line kernel. -/
def execution {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    {t : State α // exec M p Bool.false s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp (config := { maxSteps := 1000000 }) [exec, kernel, original, TypedSource.kernel, TypedSource.indexCommutedKernel,
    ComputeKernel.surfaceBody, FP.Structural.run, FP.Structural.step,
    FP.Structural.evalExpr, FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.store, FP.Structural.numeric, FP.Structural.numericLt,
    FP.Structural.bop, ComputeDType.eraseDType, ofFloat, toFloat, Region.cast, Option.bind]
  try rfl

#axiomsClean execution
end
end VeriTile.Bench.Repairs.DequantizeMatmul.KernelExecution

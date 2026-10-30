import bench.tritonbench_g.quantize_kv_transform.Kernels
import bench.tritonbench_g.kcache_copy_triton.Sources.Mathematical_TypedCopy

namespace VeriTile.Bench.Repairs.QuantizeKvTransform.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 24000
set_option linter.unusedSimpArgs false

/-- Total target signed-byte conversion and equality. FP32 maximum uses
the original ordered reduction backend, retaining its input layout and axis.
Unsupported operations in rest are proved unused by the execution below. -/
def backend {α : Type} (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) : Algebra α :=
  { rest with
    floatToInt := fun p d width => if width=8 then some (fun x => some (convert x))
      else rest.floatToInt p d width
    compareEq := fun _ _ => some equal }

/-- Construct the complete successful typed source state for every payload,
including nonfinite quotients and integer casts outside the portable domain.
The selected native conversion supplies those results; no input restriction
or success premise is used. -/
def execution {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (hw : 0<p.width) :
    {t : State α // exec (backend rest convert equal) (kernel p) s=some t} := by
  refine ⟨?_,?_⟩
  swap
  simp [kernel,FP.Structural.exec,ComputeKernel.surfaceBody,
    FP.Structural.run,FP.Structural.step,FP.Structural.evalExpr,
    FP.Structural.evalComputeOp,FP.Structural.evalOp_unfold,FP.Structural.store,
    FP.Structural.numeric,FP.Structural.numericLt,FP.Structural.numericEq,
    FP.Structural.natLt,FP.Structural.bop,backend,
    ComputeDType.eraseDType,ofFloat,toFloat,Region.cast,Option.bind,
    TileShape.axisDim,hw]
  try rfl

#axiomsClean execution
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.KernelExecution

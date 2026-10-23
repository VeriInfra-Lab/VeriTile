import bench.repairs.tritonbench_g.quantize_copy_kv.Proofs.KernelSource
import bench.tritonbench_g.kcache_copy_triton.Proofs.TypedCopy

namespace VeriTile.Bench.Repairs.QuantizeCopyKv.KernelExecution
open VeriTile.Triton FP.Structural KernelSource
open VeriTile.Bench.TritonBenchG.KcacheCopyTriton.TypedCopy (masked_scatter_regs)
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 4000
set_option linter.unusedSimpArgs false

/-- A total selected signed-byte conversion and floating equality callback.
All floating arithmetic, rounding, promotions and reductions remain the
caller's actual primitives. No primitive equality law is assumed. -/
def backend {α : Type} (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) : Algebra α :=
  { rest with
    floatToInt := fun p d width => if width = 8 then some (fun x => some (convert x))
      else rest.floatToInt p d width
    compareEq := fun _ _ => some equal }

/-- Successful source execution on every admitted floating input bit pattern,
once the actual total conversion/equality callbacks have been installed. -/
def execution {α : Type} [Inhabited α] (rest : Algebra α) (convert : α → Int)
    (equal : α → α → Bool) (p : Parameters) (s : State α) (hw : 0 < p.width) :
    {t : State α // exec (backend rest convert equal) (originalKernel p) s = some t} := by
  refine ⟨?_, ?_⟩
  swap
  simp [originalKernel, kernel, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold, FP.Structural.store,
    FP.Structural.numeric, FP.Structural.numericLt, FP.Structural.numericEq,
    FP.Structural.natLt, FP.Structural.bop, backend,
    ComputeDType.eraseDType, ofFloat, toFloat, Region.cast, Option.bind,
    TileShape.axisDim, hw]
  try rfl

theorem same_run {α : Type} [Inhabited α] (M : Algebra α) (p : Parameters) (s : State α) :
    exec M (originalKernel p) s = exec M (commutedKernel p) s := by
  simp [originalKernel, commutedKernel, kernel, FP.Structural.exec, ComputeKernel.surfaceBody,
    FP.Structural.run, FP.Structural.step, FP.Structural.evalExpr,
    FP.Structural.evalComputeOp, FP.Structural.evalOp_unfold,
    FP.Structural.numeric, FP.Structural.numericLt, FP.Structural.bop,
    ComputeDType.eraseDType, Region.cast, Option.bind]

theorem successful_equivalence {α : Type} [Inhabited α] (rest : Algebra α)
    (convert : α → Int) (equal : α → α → Bool)
    (p : Parameters) (s : State α) (hw : 0 < p.width) :
    ∃ t, exec (backend rest convert equal) (originalKernel p) s = some t ∧
      exec (backend rest convert equal) (commutedKernel p) s = some t := by
  obtain ⟨t, ht⟩ := execution rest convert equal p s hw
  exact ⟨t, ht, (same_run _ p s).symm.trans ht⟩

#axiomsClean execution
#axiomsClean same_run
#axiomsClean successful_equivalence
end
end VeriTile.Bench.Repairs.QuantizeCopyKv.KernelExecution

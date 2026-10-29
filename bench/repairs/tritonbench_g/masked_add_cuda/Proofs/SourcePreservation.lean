import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.LegacySurface
import bench.repairs.tritonbench_g.masked_add_cuda.Proofs.Graph
import VeriTile.Meta.StatementAudit

/-! The previous presentation inserted an FP32 alpha cast absent from Python,
and omitted the integer mask in its inferred input-port list. The new literal
body therefore is not the same AST. Preserve the full Real execution and the
actual selected native graph, where that old same-format cast is identity. -/
namespace VeriTile.Bench.Repairs.MaskedAddCuda.SourcePreservation
open VeriTile.Triton
set_option maxRecDepth 24000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

/-- Complete Real execution, for both signed mask widths and every old size. -/
theorem kernel (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : BlockState) :
    exec (MaskedAddCuda.kernel storage g p m n a B).eraseDType.toAlgKernel s =
      exec (LegacySurface.kernel storage g p m n a B).eraseDType.toAlgKernel s := by
  rw [source_execution]
  cases storage <;> rfl

theorem commutedKernel (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : BlockState) :
    exec (MaskedAddCuda.commutedKernel storage g p m n a B).eraseDType.toAlgKernel s =
      exec (LegacySurface.commutedKernel storage g p m n a B).eraseDType.toAlgKernel s := by
  rw [commuted_execution]
  cases storage <;> rfl

/-- Full typed execution in the existing fused graph model; arbitrary source
words and state. No whole-program equivalence is supplied as a premise. -/
theorem kernel_native (rest : FP.Structural.Algebra Graph.Expr)
    (word : Graph.Word) (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : FP.Structural.State Graph.Expr) :
    FP.Structural.exec (Graph.nativeAlgebra rest word)
      (MaskedAddCuda.kernel storage g p m n a B) s =
    FP.Structural.exec (Graph.nativeAlgebra rest word)
      (LegacySurface.kernel storage g p m n a B) s := by
  cases storage
  all_goals
    simp [MaskedAddCuda.kernel,source32,source64,LegacySurface.kernel,
      FP.Structural.exec,ComputeKernel.surfaceBody,FP.Structural.run,
      FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalComputeOp,
      FP.Structural.evalOp_unfold,Option.bind,Graph.nativeAlgebra,Graph.algebra]

theorem commuted_native (rest : FP.Structural.Algebra Graph.Expr)
    (word : Graph.Word) (storage : MaskStorage) (g p m : RegionName)
    (n : Nat) (a : ℝ) (B : Nat) (s : FP.Structural.State Graph.Expr) :
    FP.Structural.exec (Graph.nativeAlgebra rest word)
      (MaskedAddCuda.commutedKernel storage g p m n a B) s =
    FP.Structural.exec (Graph.nativeAlgebra rest word)
      (LegacySurface.commutedKernel storage g p m n a B) s := by
  cases storage
  all_goals
    simp [MaskedAddCuda.commutedKernel,commuted32,commuted64,LegacySurface.commutedKernel,
      FP.Structural.exec,ComputeKernel.surfaceBody,FP.Structural.run,
      FP.Structural.step,FP.Structural.evalExpr,FP.Structural.evalComputeOp,
      FP.Structural.evalOp_unfold,Option.bind,Graph.nativeAlgebra,Graph.algebra]

#axiomsClean kernel
#axiomsClean commutedKernel
#axiomsClean kernel_native
#axiomsClean commuted_native
end VeriTile.Bench.Repairs.MaskedAddCuda.SourcePreservation

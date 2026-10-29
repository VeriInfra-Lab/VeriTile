import bench.examples.HyperConnectionsDepth.IO
import bench.examples.HyperConnectionsDepth.Proofs.MathematicalCorrect
import bench.examples.HyperConnectionsDepth.Proofs.NativeProjection
import VeriTile.Triton.Math.Sinkhorn
import VeriTile.Triton.Math.MatrixRewrite
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.HyperConnectionsDepth
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ (Implements)
open scoped VeriTile.Triton.KernelIO₃

/-- Scalar specialization S = T = D = 1: per-program residual/branch cells and a shared logit. -/
@[reducible] def mhcDepthIO (tau : ℝ) : KernelIO₃ :=
  IO.scalarLayout (Kernels.mhcDepthConnectionKernel "res_mix" "branch_out" "h_post" "out" 1 1 1 0 tau)
    (by rfl)

/-- The scalar depth kernel implements residual + exp(logit / tau) * branch. -/
specification mhc_depth_correctness (tau : ℝ) :
    Spec.Real (mhcDepthIO tau ⊨ fun resMix branchOut hPost _ =>
      resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0) := by
  simpa only [KernelIO₃.Implements, mhcDepthIO, Mathematical.mhcDepthIO, IO.scalarLayout, NativeProjection.mhcDepthConnectionKernel_projection] using
    Mathematical.mhc_depth_correctness tau

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
@[reducible] def optimizedIO (tau : ℝ) : KernelIO₃ :=
  IO.scalarLayout (Kernels.optimizedKernel tau) (by rfl)

/-- The optimized implementation computes the same independent real formula. -/
specification mhc_depth_optimized_correctness (tau : ℝ) :
    Spec.Real (optimizedIO tau ⊨ fun resMix branchOut hPost _ =>
      resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0) := by
  simpa only [KernelIO₃.Implements, optimizedIO, Mathematical.optimizedIO, IO.scalarLayout, NativeProjection.optimizedKernel_projection] using
    Mathematical.mhc_depth_optimized_correctness tau

#axiomsClean mhc_depth_optimized_correctness
#stmtSurfaceSubset mhc_depth_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.KernelIO₃.Implements,
   VeriTile.Triton.KernelIO₃.B1, VeriTile.Triton.KernelIO₃.B2,
   VeriTile.Triton.KernelIO₃.B3, VeriTile.Triton.KernelIO₃.Bout]

end Optimized

#axiomsClean mhc_depth_correctness
#stmtSurfaceSubset mhc_depth_correctness ⊆
  [Spec.Real, mhcDepthIO, VeriTile.Triton.KernelIO₃.Implements,
   VeriTile.Triton.KernelIO₃.B1, VeriTile.Triton.KernelIO₃.B2,
   VeriTile.Triton.KernelIO₃.B3, VeriTile.Triton.KernelIO₃.Bout]

end VeriTile.Bench.Examples.HyperConnectionsDepth

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect
open VeriTile Triton LogSinkhorn
open VeriTile.Bench.Examples.HyperConnectionsDepth.Kernels

set_option maxHeartbeats 3200000

noncomputable def formula (res : Tile .real [S, D]) (branch : Tile .real [T, D])
    (logits : Tile .real [T, S]) (tau : ℝ) (iters : Nat) : Tile .real [S, D] :=
  Tile.bop WithBot.realAdd (.consSame (.consSame .nil)) res
    (Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) branch)

specification mhc_depth_matrix_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "out" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((formula (readMatrix s "res_mix" (s.pid * (S * D)) S D)
          (readMatrix s "branch_out" (s.pid * (T * D)) T D)
          (readMatrix s "h_post" 0 T S) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : TileIndex [S, D],
        o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) → t.mem r o = s.mem r o)) := by
  simpa only [NativeProjection.matrixOriginal_projection, formula, Mathematical.MatrixCorrect.formula] using
    Mathematical.MatrixCorrect.mhc_depth_matrix_correct S T D iters tau s

#axiomsClean mhc_depth_matrix_correct

specification mhc_depth_matrix_optimized_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "out" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((formula (readMatrix s "res_mix" (s.pid * (S * D)) S D)
          (readMatrix s "branch_out" (s.pid * (T * D)) T D)
          (readMatrix s "h_post" 0 T S) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : TileIndex [S, D],
        o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) → t.mem r o = s.mem r o)) := by
  simpa only [NativeProjection.matrixOptimized_projection, formula, Mathematical.MatrixCorrect.formula] using
    Mathematical.MatrixCorrect.mhc_depth_matrix_optimized_correct S T D iters tau s

#axiomsClean mhc_depth_matrix_optimized_correct

open scoped VeriTile.Triton.KernelIO₃

/-- Row-major input/output windows for either complete matrix source. -/
def matrixIO (optimized : Bool) (S T D iters : Nat) (tau : ℝ) : KernelIO₃ :=
  IO.matrixLayout
    (if optimized then (Kernels.matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau)
     else (Kernels.matrixOriginal S T D iters tau))
    (by cases optimized <;> rfl) S T D

/-- Successful pointer-kernel execution, independent matrix formula and frame,
for every disjoint placement whose buffers cover the declared windows. -/
specification mhc_depth_matrix_flat_correctness (optimized : Bool)
    (S T D iters : Nat) (tau : ℝ) :
    Spec.Real (matrixIO optimized S T D iters tau ⊨ fun res branch logits i =>
      ((formula (matrixTile res) (matrixTile branch) (matrixTile logits) tau iters).data
        (matrixIndex i)).unbotD 0) := by
  simpa only [KernelIO₃.Implements, matrixIO, Mathematical.MatrixCorrect.matrixIO, IO.matrixLayout, apply_ite, NativeProjection.matrixOriginal_projection, NativeProjection.matrixOptimized_projection, formula, Mathematical.MatrixCorrect.formula] using
    Mathematical.MatrixCorrect.mhc_depth_matrix_flat_correctness optimized S T D iters tau

#axiomsClean mhc_depth_matrix_flat_correctness

end VeriTile.Bench.Examples.HyperConnectionsDepth.MatrixCorrect

import bench.examples.HyperConnectionsWidth.IO
import bench.examples.HyperConnectionsWidth.Proofs.Real
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

namespace VeriTile.Bench.Examples.HyperConnectionsWidth.Mathematical
open VeriTile.Bench.Examples.HyperConnectionsWidth.MathKernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ₓ₂ (Implements)
open scoped VeriTile.Triton.KernelIO₃ₓ₂

/-- Scalar specialization S = T = D = 1: a per-program residual, shared logits and two output cells. -/
@[reducible] def mhcWidthIO (tau : ℝ) : KernelIO₃ₓ₂ :=
  IO.scalarLayout (MathKernels.mhcWidthConnectionKernel "res" "h_res" "h_pre" "res_mix" "branch_in" 1 1 1 0 tau)
    (by rfl)

/-- The scalar width kernel computes the residual mix and branch input independently. -/
specification mhc_width_correctness (tau : ℝ) :
    Spec.Real (mhcWidthIO tau ⊨ fun res hRes hPre =>
      (fun _ => Real.exp (hRes 0 / tau) * res 0,
       fun _ => Real.exp (hPre 0 / tau) * res 0)) := by
  refine KernelIO₃ₓ₂.Implements.intro _ ?_ ?_ ?_
  · exact mhcWidth_flattenOk tau
  · intro bounds s h1 h2 h3 h4 h5
    exact mhcWidth_traceSafe tau bounds s h1 h2 h3 h4 h5
  · intro s₀ res hRes hPre h1 h2 h3
    exact mhcWidth_region_run tau s₀ res hRes hPre h1 h2 h3

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
@[reducible] def optimizedIO (tau : ℝ) : KernelIO₃ₓ₂ :=
  IO.scalarLayout (MathKernels.optimizedKernel tau) (by rfl)

/-- The optimized implementation computes the same independent real formula. -/
specification mhc_width_optimized_correctness (tau : ℝ) :
    Spec.Real (optimizedIO tau ⊨ fun res hRes hPre =>
      (fun _ => Real.exp (hRes 0 / tau) * res 0,
       fun _ => Real.exp (hPre 0 / tau) * res 0)) := by
  refine KernelIO₃ₓ₂.Implements.intro _
    ?_ ?_ ?_
  · simp [optimizedIO, IO.scalarLayout, optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3 h4 h5
    simpa [optimizedIO, mhcWidthIO, IO.scalarLayout, Kernel.TraceSafe, optimizedKernel, mhcWidthConnectionKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, Tile.uop, NumericDType.mul, NumericDType.div, mul_comm]
      using mhcWidth_traceSafe tau bounds s h1 h2 h3 h4 h5
  · intro s res hRes hPre h1 h2 h3
    change ∃ s1, exec (optimizedKernel tau).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    exact mhcWidth_region_run tau s res hRes hPre h1 h2 h3

#axiomsClean mhc_width_optimized_correctness
#stmtSurfaceSubset mhc_width_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.KernelIO₃ₓ₂.Implements,
   VeriTile.Triton.KernelIO₃ₓ₂.B1, VeriTile.Triton.KernelIO₃ₓ₂.B2,
   VeriTile.Triton.KernelIO₃ₓ₂.B3, VeriTile.Triton.KernelIO₃ₓ₂.Bout1,
   VeriTile.Triton.KernelIO₃ₓ₂.Bout2]

end Optimized

#axiomsClean mhc_width_correctness
#stmtSurfaceSubset mhc_width_correctness ⊆
  [Spec.Real, mhcWidthIO, VeriTile.Triton.KernelIO₃ₓ₂.Implements,
   VeriTile.Triton.KernelIO₃ₓ₂.B1, VeriTile.Triton.KernelIO₃ₓ₂.B2,
   VeriTile.Triton.KernelIO₃ₓ₂.B3, VeriTile.Triton.KernelIO₃ₓ₂.Bout1,
   VeriTile.Triton.KernelIO₃ₓ₂.Bout2]

end VeriTile.Bench.Examples.HyperConnectionsWidth.Mathematical

namespace VeriTile.Bench.Examples.HyperConnectionsWidth.Mathematical.MatrixCorrect
open VeriTile Triton LogSinkhorn
open VeriTile.Bench.Examples.HyperConnectionsWidth.MathKernels

set_option maxHeartbeats 6400000

set_option linter.unusedSimpArgs false

noncomputable def residualFormula (res : Tile .real [S, D]) (logits : Tile .real [S, S])
    (tau : ℝ) (iters : Nat) : Tile .real [S, D] := Tile.dot [] (weights (scale logits tau) iters) res

noncomputable def branchFormula (res : Tile .real [S, D]) (logits : Tile .real [S, T])
    (tau : ℝ) (iters : Nat) : Tile .real [T, D] :=
  Tile.dot [] (Tile.transpose [] (weights (scale logits tau) iters)) res

specification mhc_width_matrix_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOriginal S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "res_mix" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((residualFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_res" 0 S S) tau iters).data i).unbotD 0) ∧
      (∀ i : TileIndex [T, D], t.readMem "branch_in" (s.pid * (T * D) + i.1.val * D + i.2.1.val) =
        ((branchFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_pre" 0 S T) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "res_mix" ∨ ∀ i : TileIndex [S, D], o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) →
        (r ≠ "branch_in" ∨ ∀ i : TileIndex [T, D], o ≠ s.pid * (T * D) + i.1.val * D + i.2.1.val) →
        t.mem r o = s.mem r o)) := by
  exact matrix_original_run S T D iters tau s

#axiomsClean mhc_width_matrix_correct

specification mhc_width_matrix_optimized_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "res_mix" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((residualFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_res" 0 S S) tau iters).data i).unbotD 0) ∧
      (∀ i : TileIndex [T, D], t.readMem "branch_in" (s.pid * (T * D) + i.1.val * D + i.2.1.val) =
        ((branchFormula (readMatrix s "res" (s.pid * (S * D)) S D)
          (readMatrix s "h_pre" 0 S T) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o,
        (r ≠ "res_mix" ∨ ∀ i : TileIndex [S, D], o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) →
        (r ≠ "branch_in" ∨ ∀ i : TileIndex [T, D], o ≠ s.pid * (T * D) + i.1.val * D + i.2.1.val) →
        t.mem r o = s.mem r o)) := by
  exact matrix_optimized_run S T D iters tau s

#axiomsClean mhc_width_matrix_optimized_correct

open scoped VeriTile.Triton.KernelIO₃ₓ₂

/-- Both output matrices have their own bounded row-major windows. -/
def matrixIO (optimized : Bool) (S T D iters : Nat) (tau : ℝ) : KernelIO₃ₓ₂ :=
  IO.matrixLayout
    (if optimized then (MathKernels.matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau)
     else (MathKernels.matrixOriginal S T D iters tau))
    (by cases optimized <;> rfl) S T D

/-- The complete pointer kernel computes both independent matrix formulas and
preserves every flat cell outside their output windows. -/
specification mhc_width_matrix_flat_correctness (optimized : Bool)
    (S T D iters : Nat) (tau : ℝ) :
    Spec.Real (matrixIO optimized S T D iters tau ⊨ fun res hres hpre =>
      (fun i => ((residualFormula (matrixTile res) (matrixTile hres) tau iters).data
        (matrixIndex i)).unbotD 0,
       fun i => ((branchFormula (matrixTile res) (matrixTile hpre) tau iters).data
        (matrixIndex i)).unbotD 0)) := by
  refine KernelIO₃ₓ₂.Implements.intro _ ?_ ?_ ?_
  · simpa only [matrixIO, IO.matrixLayout, apply_ite] using Memory.flattenOk optimized S T D iters tau
  · intro bounds s h1 h2 h3 h4 h5
    simp only [matrixIO, IO.matrixLayout, Nat.zero_add] at h1 h2 h3 h4 h5
    simpa only [matrixIO, IO.matrixLayout, apply_ite] using Memory.traceSafe optimized S T D iters tau bounds s h1 h2 h3 h4 h5
  · intro s xs ys zs hx hy hz
    simpa only [matrixIO, IO.matrixLayout, apply_ite, residualFormula, branchFormula] using
      matrix_region_run optimized S T D iters tau s xs ys zs hx hy hz

#axiomsClean mhc_width_matrix_flat_correctness

end VeriTile.Bench.Examples.HyperConnectionsWidth.Mathematical.MatrixCorrect

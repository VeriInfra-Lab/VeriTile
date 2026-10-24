import bench.examples.HyperConnectionsDepth.IO
import bench.examples.HyperConnectionsDepth.Proofs.Real
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

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.Mathematical
open VeriTile.Bench.Examples.HyperConnectionsDepth.MathKernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Triton
open VeriTile.Triton.KernelIO₃ (Implements)
open scoped VeriTile.Triton.KernelIO₃

/-- Scalar specialization S = T = D = 1: per-program residual/branch cells and a shared logit. -/
@[reducible] def mhcDepthIO (tau : ℝ) : KernelIO₃ :=
  IO.scalarLayout (MathKernels.mhcDepthConnectionKernel "res_mix" "branch_out" "h_post" "out" 1 1 1 0 tau)
    (by rfl)

/-- The scalar depth kernel implements residual + exp(logit / tau) * branch. -/
specification mhc_depth_correctness (tau : ℝ) :
    Spec.Real (mhcDepthIO tau ⊨ fun resMix branchOut hPost _ =>
      resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0) := by
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · exact mhcDepth_flattenOk tau
  · intro bounds s h1 h2 h3 h4 _
    exact mhcDepth_traceSafe tau bounds s h1 h2 h3 h4
  · intro s₀ resMix branchOut hPost h1 h2 h3
    obtain ⟨s1, hexec, hval, hframe⟩ :=
      mhcDepth_region_run tau s₀ resMix branchOut hPost h1 h2 h3
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
@[reducible] def optimizedIO (tau : ℝ) : KernelIO₃ :=
  IO.scalarLayout (MathKernels.optimizedKernel tau) (by rfl)

/-- The optimized implementation computes the same independent real formula. -/
specification mhc_depth_optimized_correctness (tau : ℝ) :
    Spec.Real (optimizedIO tau ⊨ fun resMix branchOut hPost _ =>
      resMix 0 + Real.exp (hPost 0 / tau) * branchOut 0) := by
  refine KernelIO₃.Implements.intro _
    ?_ ?_ ?_
  · simp [optimizedIO, IO.scalarLayout, optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3 h4 _
    simpa [optimizedIO, mhcDepthIO, IO.scalarLayout, Kernel.TraceSafe, optimizedKernel, mhcDepthConnectionKernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Region.cast, Tile.bop, Tile.uop, NumericDType.add, NumericDType.mul,
      NumericDType.div, add_comm]
      using mhcDepth_traceSafe tau bounds s h1 h2 h3 h4
  · intro s resMix branchOut hPost h1 h2 h3
    change ∃ s1, exec (optimizedKernel tau).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    obtain ⟨s1, he, hv, hf⟩ := mhcDepth_region_run tau s resMix branchOut hPost h1 h2 h3
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

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

end VeriTile.Bench.Examples.HyperConnectionsDepth.Mathematical

namespace VeriTile.Bench.Examples.HyperConnectionsDepth.Mathematical.MatrixCorrect
open VeriTile Triton LogSinkhorn
open VeriTile.Bench.Examples.HyperConnectionsDepth.MathKernels

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
  exact matrix_original_run S T D iters tau s

#axiomsClean mhc_depth_matrix_correct

specification mhc_depth_matrix_optimized_correct (S T D iters : Nat) (tau : ℝ) (s : BlockState) :
    Spec.Real (∃ t, exec (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau) s = some t ∧
      (∀ i : TileIndex [S, D], t.readMem "out" (s.pid * (S * D) + i.1.val * D + i.2.1.val) =
        ((formula (readMatrix s "res_mix" (s.pid * (S * D)) S D)
          (readMatrix s "branch_out" (s.pid * (T * D)) T D)
          (readMatrix s "h_post" 0 T S) tau iters).data i).unbotD 0) ∧
      (∀ (r : RegionName) o, (r ≠ "out" ∨ ∀ i : TileIndex [S, D],
        o ≠ s.pid * (S * D) + i.1.val * D + i.2.1.val) → t.mem r o = s.mem r o)) := by
  exact matrix_optimized_run S T D iters tau s

#axiomsClean mhc_depth_matrix_optimized_correct

open scoped VeriTile.Triton.KernelIO₃

/-- Row-major input/output windows for either complete matrix source. -/
def matrixIO (optimized : Bool) (S T D iters : Nat) (tau : ℝ) : KernelIO₃ :=
  IO.matrixLayout
    (if optimized then (MathKernels.matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau)
     else (MathKernels.matrixOriginal S T D iters tau))
    (by cases optimized <;> rfl) S T D

/-- Successful pointer-kernel execution, independent matrix formula and frame,
for every disjoint placement whose buffers cover the declared windows. -/
specification mhc_depth_matrix_flat_correctness (optimized : Bool)
    (S T D iters : Nat) (tau : ℝ) :
    Spec.Real (matrixIO optimized S T D iters tau ⊨ fun res branch logits i =>
      ((formula (matrixTile res) (matrixTile branch) (matrixTile logits) tau iters).data
        (matrixIndex i)).unbotD 0) := by
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · simpa only [matrixIO, IO.matrixLayout, apply_ite] using Memory.flattenOk optimized S T D iters tau
  · intro bounds s h1 h2 h3 h4 _
    simp only [matrixIO, IO.matrixLayout, Nat.zero_add] at h1 h2 h3 h4
    simpa only [matrixIO, IO.matrixLayout, apply_ite] using Memory.traceSafe optimized S T D iters tau bounds s h1 h2 h3 h4
  · intro s xs ys zs hx hy hz
    obtain ⟨t, ht, hv, hf⟩ := matrix_region_run optimized S T D iters tau s xs ys zs hx hy hz
    refine ⟨t, ?_, hv, fun r o h _ => hf r o h⟩
    simpa only [matrixIO, IO.matrixLayout, apply_ite] using ht

#axiomsClean mhc_depth_matrix_flat_correctness

end VeriTile.Bench.Examples.HyperConnectionsDepth.Mathematical.MatrixCorrect

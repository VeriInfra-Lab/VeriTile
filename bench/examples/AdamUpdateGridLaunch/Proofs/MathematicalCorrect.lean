import bench.examples.AdamUpdateGridLaunch.IO
import bench.examples.AdamUpdateGridLaunch.Proofs.Real
import VeriTile.Examples.Common
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math.Optimizer
import VeriTile.Triton.Launch.Composition
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch.Mathematical
open VeriTile.Bench.Examples.AdamUpdateGridLaunch.MathKernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Examples
open VeriTile.Triton

set_option maxHeartbeats 5000000

open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

/-- Masked in-place IO: the parameter and momentum buffers are both read and updated. -/
def adamIO (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ :=
  IO.layout (MathKernels.update_fn_kernel "p" "grad" "exp_avg" lr wd beta1 beta2 n B)
    (by rfl) n B

/-- Each active lane implements the Lion parameter and momentum formulas on the original inputs. -/
specification adam_update_correctness (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    Spec.Real (adamIO lr wd beta1 beta2 n B ⊨ fun p grad expAvg =>
      (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
       fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2)) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [adamIO, IO.layout]) (by simp [adamIO, IO.layout]) ?_ ?_ ?_
  · exact update_fn_kernel_flattenOk ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B
  · intro bounds s h1 h2 h3 _ _
    exact update_fn_kernel_traceSafe ⟨"p"⟩ ⟨"grad"⟩ ⟨"exp_avg"⟩
      lr wd beta1 beta2 n B bounds s h1 h2 h3
  · intro s₀ xs ys zs hx hy hz
    exact update_fn_kernel_region_run lr wd beta1 beta2 n B s₀ xs ys zs hx hy hz

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ :=
  IO.layout (MathKernels.optimizedKernel lr wd beta1 beta2 n B) (by rfl) n B

/-- The optimized implementation computes the same independent real formula. -/
specification adam_update_optimized_correctness (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    Spec.Real (optimizedIO lr wd beta1 beta2 n B ⊨ fun p grad expAvg =>
      (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
       fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2)) := by
  refine MaskedKernelIO₃ₓ₂.Implements.intro _
    (by simp [optimizedIO, adamIO, IO.layout]) (by simp [optimizedIO, adamIO, IO.layout]) ?_ ?_ ?_
  · simp [optimizedIO, IO.layout, optimizedKernel, Kernel.FlattenOk,
      StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]
  · intro bounds s h1 h2 h3 _ _
    simpa [optimizedIO, adamIO, IO.layout, Kernel.TraceSafe, optimizedKernel, update_fn_kernel,
      Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt,
      MemAccess.SafeAt, stepStmt, evalOp.eq_def,
      Tile.bop, Tile.cop, Tile.select, Tile.ptrAdd,
      NumericDType.add, NumericDType.mul, NumericDType.sub,
      ComparableDType.lt, ComparableDType.gt, ComparableDType.ne, add_comm, optional_add_comm]
      using update_fn_kernel_traceSafe "p" "grad" "exp_avg" lr wd beta1 beta2 n B bounds s h1 h2 h3
  · intro s xs ys zs hx hy hz
    change ∃ s1, exec (optimizedKernel lr wd beta1 beta2 n B).toAlgKernel s = some s1 ∧ _
    rw [optimized_exec]
    exact update_fn_kernel_region_run lr wd beta1 beta2 n B s xs ys zs hx hy hz

#axiomsClean adam_update_optimized_correctness
#stmtSurfaceSubset adam_update_optimized_correctness ⊆
  [Spec.Real, optimizedIO, VeriTile.Triton.MaskedKernelIO₃ₓ₂.Implements,
   TiledOptimizer.lionParam, TiledOptimizer.lionMomentum,
   VeriTile.Triton.MaskedKernelIO₃ₓ₂.B]

end Optimized

#axiomsClean adam_update_correctness
#stmtSurfaceSubset adam_update_correctness ⊆
  [Spec.Real, adamIO, VeriTile.Triton.MaskedKernelIO₃ₓ₂.Implements,
   TiledOptimizer.lionParam, TiledOptimizer.lionMomentum,
   VeriTile.Triton.MaskedKernelIO₃ₓ₂.B]

end VeriTile.Bench.Examples.AdamUpdateGridLaunch.Mathematical

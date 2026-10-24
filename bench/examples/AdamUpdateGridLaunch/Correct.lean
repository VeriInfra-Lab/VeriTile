import bench.examples.AdamUpdateGridLaunch.IO
import bench.examples.AdamUpdateGridLaunch.Proofs.MathematicalCorrect
import bench.examples.AdamUpdateGridLaunch.Proofs.NativeProjection
import bench.examples.AdamUpdateGridLaunch.Proofs.Grid
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

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch
open VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels

attribute [local simp] VeriTile.Triton.ComputeExpr.toAlgorithm?
  VeriTile.Triton.ComputeOp.toAlgorithm? VeriTile.Triton.ComputeDType.eraseDType

open VeriTile.Examples
open VeriTile.Triton

set_option maxHeartbeats 5000000

open scoped VeriTile.Triton.MaskedKernelIO₃ₓ₂

/-- Masked in-place IO: the parameter and momentum buffers are both read and updated. -/
def adamIO (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ :=
  IO.layout (Kernels.update_fn_kernel "p" "grad" "exp_avg" lr wd beta1 beta2 n B)
    (by rfl) n B

/-- Each active lane implements the Lion parameter and momentum formulas on the original inputs. -/
specification adam_update_correctness (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    Spec.Real (adamIO lr wd beta1 beta2 n B ⊨ fun p grad expAvg =>
      (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
       fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2)) := by
  simpa only [MaskedKernelIO₃ₓ₂.Implements, adamIO, Mathematical.adamIO, IO.layout,
    NativeProjection.update_fn_kernel_projection] using
      Mathematical.adam_update_correctness lr wd beta1 beta2 n B

section Optimized

set_option maxHeartbeats 5000000

attribute [local simp] ComputeExpr.toAlgorithm? ComputeOp.toAlgorithm? ComputeDType.eraseDType

/-- The optimized source retains the original IO windows and memory contract. -/
def optimizedIO (lr wd beta1 beta2 : ℝ) (n B : Nat) : MaskedKernelIO₃ₓ₂ :=
  IO.layout (Kernels.optimizedKernel lr wd beta1 beta2 n B) (by rfl) n B

/-- The optimized implementation computes the same independent real formula. -/
specification adam_update_optimized_correctness (lr wd beta1 beta2 : ℝ) (n B : Nat) :
    Spec.Real (optimizedIO lr wd beta1 beta2 n B ⊨ fun p grad expAvg =>
      (fun i => TiledOptimizer.lionParam (p i) (expAvg i) (grad i) lr wd beta1,
       fun i => TiledOptimizer.lionMomentum (expAvg i) (grad i) beta2)) := by
  simpa only [MaskedKernelIO₃ₓ₂.Implements, optimizedIO, Mathematical.optimizedIO, IO.layout,
    NativeProjection.optimizedKernel_projection] using
      Mathematical.adam_update_optimized_correctness lr wd beta1 beta2 n B

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

end VeriTile.Bench.Examples.AdamUpdateGridLaunch

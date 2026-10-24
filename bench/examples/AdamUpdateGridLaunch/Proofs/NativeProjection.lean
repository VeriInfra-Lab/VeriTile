import bench.examples.AdamUpdateGridLaunch.Kernels
import bench.examples.AdamUpdateGridLaunch.Proofs.MathKernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch.NativeProjection
open VeriTile Triton
set_option maxHeartbeats 6400000
set_option linter.unusedSimpArgs false

@[simp] theorem update_fn_kernel_projection (p_ptr grad_ptr exp_avg_ptr : RegionName) (lr wd beta1 beta2 : ℝ) (n B : Nat) :
  (Kernels.update_fn_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B).toAlgKernel =
    (MathKernels.update_fn_kernel p_ptr grad_ptr exp_avg_ptr lr wd beta1 beta2 n B).toAlgKernel := by
  rfl
#axiomsClean update_fn_kernel_projection

@[simp] theorem optimizedKernel_projection (lr wd beta1 beta2 : ℝ) (n B : Nat) :
  (Kernels.optimizedKernel lr wd beta1 beta2 n B).toAlgKernel =
    (MathKernels.optimizedKernel lr wd beta1 beta2 n B).toAlgKernel := by
  rfl
#axiomsClean optimizedKernel_projection
end VeriTile.Bench.Examples.AdamUpdateGridLaunch.NativeProjection

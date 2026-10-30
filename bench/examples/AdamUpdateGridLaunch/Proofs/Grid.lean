import bench.examples.AdamUpdateGridLaunch.Proofs.Real
import bench.examples.AdamUpdateGridLaunch.Proofs.NativeProjection
import VeriTile.Meta.StatementAudit

/-! Complete native mathematical grid launch. The full-kernel projection
transports the constructed ordinary launch, all output cells and memory
framing; the scope is not reduced to a single tile. -/
namespace VeriTile.Bench.Examples.AdamUpdateGridLaunch
open VeriTile Triton
open VeriTile.Examples
open VeriTile.Bench.Examples.AdamUpdateGridLaunch.Kernels


theorem update_fn_kernel_launchCorrect
    (p_ptr grad_ptr exp_avg_ptr : RegionName)
    (lr wd beta1 beta2 : ℝ) (n_elements BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) (hRegions : p_ptr ≠ exp_avg_ptr) (s : BlockState) :
    Kernel.LaunchCorrect
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel
      (adamGrid ((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE)) s
      (fun k : Nat => if k < n_elements then some (p_ptr, k) else none)
      (fun k => TiledOptimizer.lionParam (s.readMem p_ptr k)
        (s.readMem exp_avg_ptr k) (s.readMem grad_ptr k) lr wd beta1) ∧
    Kernel.LaunchCorrect
      (update_fn_kernel p_ptr grad_ptr exp_avg_ptr
        lr wd beta1 beta2 n_elements BLOCK_SIZE).toAlgKernel
      (adamGrid ((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE)) s
      (fun k : Nat => if k < n_elements then some (exp_avg_ptr, k) else none)
      (fun k => TiledOptimizer.lionMomentum (s.readMem exp_avg_ptr k)
        (s.readMem grad_ptr k) beta2) := by
  simpa only [NativeProjection.update_fn_kernel_projection] using
    Mathematical.update_fn_kernel_launchCorrect p_ptr grad_ptr exp_avg_ptr
      lr wd beta1 beta2 n_elements BLOCK_SIZE hB hRegions s

#axiomsClean update_fn_kernel_launchCorrect
end VeriTile.Bench.Examples.AdamUpdateGridLaunch

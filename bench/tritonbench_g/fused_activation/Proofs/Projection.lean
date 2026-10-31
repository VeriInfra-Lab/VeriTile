import bench.tritonbench_g.fused_activation.Kernels
import bench.tritonbench_g.fused_activation.Proofs.Mathematical_Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.FusedActivation.Projection
open VeriTile.Triton

theorem original (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) :
    (kernel x b i W N B m a).toAlgKernel =
      (VeriTile.Bench.TritonBenchG.FusedActivation.fused_add_mul_activation_kernel
        x b i W N B m a).toAlgKernel := by cases a <;> rfl

theorem original_projection (x b i : RegionName) (W N B : Nat) (m : ℝ) (a : Bool) :
    (kernel x b i W N B m a).toAlgorithm? = .ok (kernel x b i W N B m a).toAlgKernel := by
  cases a <;> rfl

#axiomsClean original
#axiomsClean original_projection
end VeriTile.Bench.Repairs.FusedActivation.Projection

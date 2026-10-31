import bench.tritonbench_g.fused_activation.Kernels
import VeriTile.Triton.DSL

/-! Complete masked in-place fp32 source from fused_activation.py and a candidate
commuting its final bias addition. Both sigmoid and ReLU branches are retained.
The source has no explicit casts; FPEquiv selects the fp32 arithmetic profile.
The host launch is outside these per-program contracts. -/

namespace VeriTile.Bench.TritonBenchG.FusedActivation
open VeriTile.Triton

/-- Mathematical view derived from the shared repair quotation. -/
noncomputable def fused_add_mul_activation_kernel
    (x_ptr bias_ptr in_ptr : RegionName)
    (num_weights xnumel BLOCK_SIZE : Nat)
    (multiplier : ℝ) (activation : Bool) :
    ComputeKernel :=
  let k := (VeriTile.Bench.Repairs.FusedActivation.kernel x_ptr bias_ptr in_ptr num_weights xnumel BLOCK_SIZE multiplier activation).toAlgKernel
  ComputeKernel.fromKernelBody k.inputs k.outputs k.body

end VeriTile.Bench.TritonBenchG.FusedActivation

import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelCheck
import bench.tritonbench_g.fused_recurrent_delta.Proofs.KernelScale

/-! Kernel-entry admission over the same shared Tensor metadata used by the
source/Real/native storage endpoints. The historical wrapper contract lives
in Misc/HostContract.lean and is not imported by these kernel endpoints. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
abbrev kernel_check := KernelCheck.checkInput
abbrev kernel_checked := @KernelCheck.checkInput_iff
abbrev kernel_scale_projection := @KernelScale.finite_projection
#axiomsClean kernel_checked
#axiomsClean kernel_scale_projection
end VeriTile.Bench.Repairs.FusedRecurrentDelta

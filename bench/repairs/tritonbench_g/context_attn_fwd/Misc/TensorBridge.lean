import bench.repairs.tritonbench_g.context_attn_fwd.Misc.KernelOnly
import bench.repairs.tritonbench_g.context_attn_fwd.Proofs.RawCompatibility

/-! Optional input-Tensor binding exported separately from the complete native result. -/
namespace VeriTile.Bench.Repairs.ContextAttnFwd.TensorBridge
alias input_tensor := KernelOnly.input_tensor
alias correct_real := KernelOnly.mathematical_real
alias correct_fp32 := KernelOnly.native_tensor
alias raw_real := RawCompatibility.correct_real
alias raw_real_grid := RawCompatibility.correct_real_grid
alias raw_fp32 := RawCompatibility.correct_fp32
alias raw_fp32_grid := RawCompatibility.correct_fp32_grid
#axiomsClean correct_real
#axiomsClean correct_fp32
#axiomsClean input_tensor
end VeriTile.Bench.Repairs.ContextAttnFwd.TensorBridge

import bench.repairs.tritonbench_g.fused_recurrent_delta.Contract
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelRealSaved
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelIndex
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelCompiler
import bench.repairs.tritonbench_g.fused_recurrent_delta.Proofs.KernelCoverage
/-! Existing raw execution and geometry endpoints keep their exact declarations.
They share the source-execution foundation with the public IO theorems. -/

/-! Complete raw-state source execution, mathematical forward/VJP values and frame. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
open VeriTile.Triton FP.Structural KernelProgram KernelTensor
open VeriTile.Bench.Repairs.LayernormFwdTriton
noncomputable section

/- Complete raw forward source, independent mathematical values and frame. -/
export KernelReal (forward_complete backward_complete)

abbrev kernel_forward_programs := @KernelReal.forward_entries_execute
abbrev kernel_saved_from_forward := @KernelReal.saved_from_forward
abbrev kernel_key_padding := @KernelReal.pad_gradient
abbrev kernel_full_tensor_vjp := @Math.tensor_loss_derivative
abbrev kernel_scalar_beta_vjp := @Math.scalar_beta_vjp

#axiomsClean forward_complete
#axiomsClean backward_complete
#axiomsClean kernel_forward_programs
#axiomsClean kernel_saved_from_forward
#axiomsClean kernel_full_tensor_vjp
end
end VeriTile.Bench.Repairs.FusedRecurrentDelta


/-! Exact native words for both complete source kernels over raw state, all
stores, and complete regional frames.
Compiler decoding and local ISA operations are explicit model data. No
primitive=software or whole-kernel numerical-success premise is present. -/
namespace VeriTile.Bench.Repairs.FusedRecurrentDelta
abbrev kernel_native_source := @KernelCompiler.complete
abbrev kernel_runtime_scale := @KernelCompiler.runtime_scale
abbrev kernel_positive_underflow := @KernelCompiler.positive_underflow_admitted
abbrev kernel_pointer_projection := @KernelIndex.active_exact
abbrev kernel_formed_indices := @KernelIndex.formed_i32
abbrev kernel_reverse_retreat := @KernelIndex.reverse_retreat
abbrev kernel_program_separation := @KernelGrid.unique_program
abbrev kernel_forward_coverage := @KernelCoverage.forward_values
abbrev kernel_forward_state_coverage := @KernelCoverage.forward_state
abbrev kernel_backward_key_coverage := @KernelCoverage.backward_keys
abbrev kernel_backward_value_coverage := @KernelCoverage.backward_value
abbrev kernel_backward_state_coverage := @KernelCoverage.backward_state
abbrev kernel_scalar_beta_coverage := @KernelCoverage.scalar_beta
abbrev kernel_vector_beta_coverage := @KernelCoverage.vector_beta

#axiomsClean kernel_native_source
#axiomsClean kernel_program_separation
#specNonCircular KernelTensor.Buffers avoiding [VeriTile.Triton.FP.Structural.exec,VeriTile.Triton.FP.Structural.run]
end VeriTile.Bench.Repairs.FusedRecurrentDelta

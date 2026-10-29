import bench.repairs.tritonbench_g.quantize_kv_transform.Correct
import bench.repairs.tritonbench_g.quantize_kv_transform.Misc.NativeCorrect
import bench.repairs.tritonbench_g.quantize_kv_transform.FPEquiv
import bench.repairs.tritonbench_g.quantize_kv_transform.Proofs.KernelTensor

/-! Optional original-Tensor applications and Python admission correspondence.
The Real certificate is in Correct; the ordinary native auxiliary is retained in NativeCorrect. -/
namespace VeriTile.Bench.Repairs.QuantizeKvTransform.TensorBridge
noncomputable section

alias python_checks_iff := checkArguments_iff
alias quantize_kv_transform_kernel_real := KernelWhole.mathematical
alias quantize_kv_transform_graph_native := KernelStores.native
alias quantize_kv_transform_real := KernelTensor.mathematical
alias quantize_kv_transform_native := KernelTensor.native

#axiomsClean python_checks_iff
#axiomsClean quantize_kv_transform_kernel_real
#axiomsClean quantize_kv_transform_graph_native
#axiomsClean quantize_kv_transform_real
#axiomsClean quantize_kv_transform_native
#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.Repairs.QuantizeKvTransform.TensorBridge

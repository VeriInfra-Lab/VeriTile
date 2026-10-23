import bench.repairs.tritonbench_g.fast_ce_loss.Correct
import bench.repairs.tritonbench_g.fast_ce_loss.FPEquiv
import bench.repairs.tritonbench_g.fast_ce_loss.Misc.NativeCorrect
import bench.repairs.tritonbench_g.fast_ce_loss.Misc.TensorBridge
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelFMA

namespace VeriTile.Bench.Repairs.FastCeLoss.Audit
#axiomsClean Correct.forward_kernel
#axiomsClean TensorBridge.Real.forward
#axiomsClean Correct.chunked_kernel
#axiomsClean TensorBridge.Real.chunked
#axiomsClean Correct.backward_kernel
#axiomsClean TensorBridge.Real.backward
#axiomsClean FPEquiv.forward_kernel
#axiomsClean NativeCorrect.forward_kernel
#axiomsClean TensorBridge.Native.forward
#axiomsClean FPEquiv.chunked_kernel
#axiomsClean NativeCorrect.chunked_kernel
#axiomsClean TensorBridge.Native.chunked
#axiomsClean FPEquiv.backward_kernel
#axiomsClean NativeCorrect.backward_kernel
#axiomsClean TensorBridge.Native.backward
#axiomsClean KernelFMA.forward_softcap_fused
#axiomsClean KernelFMA.backward_probability_fused
#axiomsClean KernelExecution.ignored_gradient
#axiomsClean Contract.flat_vector
#axiomsClean Contract.flat_matrix
#axiomsClean Contract.column_mask
#axiomsClean Contract.forward_addresses
#axiomsClean Contract.backward_addresses
#axiomsClean Contract.forward_logit_leaf
#axiomsClean Contract.forward_label_bytes
#specNonCircular KernelMath.fullLSE avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelMath.rowLoss avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelMath.partialLoss avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelReal.gradient avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelProfile.forwardLSE avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelProfile.fullLoss avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelProfile.partialLoss avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelProfile.backwardLane avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#specNonCircular KernelCompiler.observe avoiding [KernelSource.forward,KernelSource.chunked,KernelSource.backward]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.FastCeLoss.Audit

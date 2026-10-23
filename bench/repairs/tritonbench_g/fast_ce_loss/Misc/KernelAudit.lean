import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelInterface
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelFMA
import bench.repairs.tritonbench_g.fast_ce_loss.Proofs.KernelPublic
import bench.repairs.tritonbench_g.fast_ce_loss.Contract

namespace VeriTile.Bench.Repairs.FastCeLoss.KernelAudit
#axiomsClean KernelInterface.Real.forward_kernel
#axiomsClean KernelPublic.Real.forward
#axiomsClean KernelInterface.Real.chunked_kernel
#axiomsClean KernelPublic.Real.chunked
#axiomsClean KernelInterface.Real.backward_kernel
#axiomsClean KernelPublic.Real.backward
#axiomsClean KernelInterface.Native.forward_kernel
#axiomsClean KernelPublic.Native.forward
#axiomsClean KernelInterface.Native.chunked_kernel
#axiomsClean KernelPublic.Native.chunked
#axiomsClean KernelInterface.Native.backward_kernel
#axiomsClean KernelPublic.Native.backward
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
end VeriTile.Bench.Repairs.FastCeLoss.KernelAudit

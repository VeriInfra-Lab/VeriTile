import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.PublicIO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Misc.NativePublicIO
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelBound
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelGrid
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.KernelTail

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelAudit
open VeriTile.Triton
#axiomsClean PublicIO.real
#axiomsClean NativePublicIO.native
#axiomsClean NativePublicIO.compiled
#axiomsClean KernelSafety.forward_source
#axiomsClean KernelSafety.backward_source
#axiomsClean KernelExecution.complete
#axiomsClean KernelReal.complete
#axiomsClean KernelNative.complete
#axiomsClean KernelCompiler.complete
#axiomsClean KernelBound.real_tensor
#axiomsClean KernelBound.native_tensor
#axiomsClean KernelBounds.writes
#axiomsClean KernelBounds.word_aligned
#axiomsClean KernelLaunch.formed
#axiomsClean KernelGrid.unique
#axiomsClean KernelTail.untouched
#specNonCircular KernelReal.expected avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected]
#specNonCircular KernelNative.expected avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected]
#specNonCircular KernelLogical.expected avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected,KernelNative.expected]
#specNonCircular KernelCompiler.Binding avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected,KernelNative.expected]
#specNonCircular KernelTensor.Buffers avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected,KernelNative.expected]
#specNonCircular KernelTensor.Outputs avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected,KernelNative.expected]
#specNonCircular KernelLaunch.Admitted avoiding [FP.Structural.exec,FP.Structural.run,KernelExecution.expected,KernelNative.expected]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkGateRecurrence.KernelAudit

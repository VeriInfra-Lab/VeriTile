import bench.repairs.tritonbench_g.chunk_retention_ops.Misc.KernelCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Misc.KernelFPEquiv

namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001
open VeriTile Triton FP.Structural

#check KernelCorrect.forward_tensor
#check KernelCorrect.output_tensor
#check KernelCorrect.backward_tensor
#check KernelCorrect.gradient_tensor
#check KernelFPEquiv.forward_tensor
#check KernelFPEquiv.output_tensor
#check KernelFPEquiv.backward_tensor
#check KernelFPEquiv.gradient_tensor
#axiomsClean KernelCorrect.forward_tensor
#axiomsClean KernelCorrect.output_tensor
#axiomsClean KernelCorrect.backward_tensor
#axiomsClean KernelCorrect.gradient_tensor
#axiomsClean KernelFPEquiv.forward_tensor
#axiomsClean KernelFPEquiv.output_tensor
#axiomsClean KernelFPEquiv.backward_tensor
#axiomsClean KernelFPEquiv.gradient_tensor

#check KernelPacket.complete
#check KernelTensor.frame_of_exec
#check KernelTensor.footprint_bound
#check KernelTensor.input_frame
#axiomsClean KernelPacket.complete
#axiomsClean KernelTensor.source_tensor
#axiomsClean KernelTensor.native_complete
#axiomsClean KernelTensor.frame_of_exec
#axiomsClean KernelTensor.footprint_bound
#axiomsClean KernelTensor.coordinates
#axiomsClean KernelTensor.input_frame
#axiomsClean KernelNative.forward_target
#axiomsClean KernelNative.output_target
#axiomsClean KernelNative.backward_target
#axiomsClean KernelNative.gradient_target
#axiomsClean KernelNative.key_result_dot
#axiomsClean KernelNative.key_result_eq
#axiomsClean KernelNative.reference_laws

#specNonCircular KernelPacket.reference avoiding [run,FP.Structural.exec,KernelPacket.source,KernelPacket.complete]
#specNonCircular KernelNative.target avoiding [run,FP.Structural.exec,KernelPacket.source,KernelNative.correct]
#specNonCircular KernelNative.data avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular KernelTensor.View avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular KernelTensor.Valid avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular KernelTensor.Frame avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular Numeric.ScalarLaws avoiding [FP.Structural.exec,KernelNative.target,KernelNative.correct]
#specNonCircular MathTarget.state avoiding [run,FP.Structural.exec,KernelPacket.reference]
#specNonCircular MathTarget.future avoiding [run,FP.Structural.exec,KernelPacket.reference]
#specNonCircular ScalarSource.outputTile avoiding [run,FP.Structural.exec,KernelPacket.reference]
#specNonCircular ScalarSource.queryTile avoiding [run,FP.Structural.exec,KernelPacket.reference]
#specNonCircular ScalarSource.keyTile avoiding [run,FP.Structural.exec,KernelPacket.reference]
#specNonCircular ScalarSource.valuePartial avoiding [run,FP.Structural.exec,KernelPacket.reference]
#specNonCircular Body.dq avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular Body.dk avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular Body.dvPartial avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#specNonCircular Reference.reverseCarry avoiding [run,FP.Structural.exec,KernelPacket.reference,KernelNative.target]
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001

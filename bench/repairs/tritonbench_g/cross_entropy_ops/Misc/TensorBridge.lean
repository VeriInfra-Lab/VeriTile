import bench.repairs.tritonbench_g.cross_entropy_ops.Correct
import bench.repairs.tritonbench_g.cross_entropy_ops.Misc.NativeCorrect
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelActual
import bench.repairs.tritonbench_g.cross_entropy_ops.Proofs.KernelNativeActual

/-! Complete Tensor applications, including actual backing bytes, logical
readback and all original output/frame observations. These use the same raw
execution lemmas exposed by the primary interfaces and the existing bindings. -/
namespace VeriTile.Bench.Repairs.CrossEntropyOps.TensorBridge
export KernelActual (real_forward real_backward native_forward native_backward)
#axiomsClean real_forward
#axiomsClean real_backward
#axiomsClean native_forward
#axiomsClean native_backward
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.CrossEntropyOps.TensorBridge

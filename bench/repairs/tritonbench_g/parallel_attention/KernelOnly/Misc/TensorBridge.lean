import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.TensorEndpoints
import bench.repairs.tritonbench_g.parallel_attention.KernelOnly.JointEndpoints

/-! Complete Tensor applications, including simultaneous common-backing outputs.
Each application uses the exposed raw NativeStage execution/target/frame result. -/
namespace VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBridge
export VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBinding (forward_tensor backward_tensor)
export VeriTile.Bench.Repairs.ParallelAttentionKernel.CommonBacking (forward_joint backward_joint)
#print CommonBacking.Outputs
#print CommonBacking.ForwardBacking
#print CommonBacking.BackwardBacking
#print CommonBacking.JointBinding
#check @forward_joint
#check @backward_joint
#axiomsClean forward_joint
#axiomsClean backward_joint
#axiomsClean CommonBacking.realize_joint
#axiomsClean CommonBacking.forward_all_stores
#axiomsClean CommonBacking.backward_all_stores
#axiomsClean CommonBacking.forward_physical_unique
#axiomsClean CommonBacking.backward_physical_unique
#axiomsClean CommonBacking.forward_empty_post
#axiomsClean CommonBacking.backward_empty_post
#specNonCircular CommonBacking.forwardFootprint avoiding [CommonBacking.jointPost,VeriTile.Triton.FP.Structural.exec,runPrograms]
#specNonCircular CommonBacking.backwardFootprint avoiding [CommonBacking.jointPost,VeriTile.Triton.FP.Structural.exec,runPrograms]
#axiomsClean forward_tensor
#axiomsClean backward_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ParallelAttentionKernel.TensorBridge

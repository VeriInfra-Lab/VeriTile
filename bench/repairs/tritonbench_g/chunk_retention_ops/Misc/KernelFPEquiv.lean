import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelNativeTensor

/-! Kernel-only native FP32 entry points. Raw Tensor bits equal the independent
source-precision scalar references under the fixed local primitive profile. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelFPEquiv
export KernelNativeTensor (forward_tensor output_tensor backward_tensor gradient_tensor)
#axiomsClean forward_tensor
#axiomsClean output_tensor
#axiomsClean backward_tensor
#axiomsClean gradient_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelFPEquiv

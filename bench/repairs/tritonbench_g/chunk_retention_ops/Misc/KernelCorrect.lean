import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelRealTensor

/-! Kernel-only exact-Real entry points. Each theorem derives successful actual
quotation execution, independent scalar results, and exact Tensor/memory frames. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelCorrect
export KernelRealTensor (forward_tensor output_tensor backward_tensor gradient_tensor)
#axiomsClean forward_tensor
#axiomsClean output_tensor
#axiomsClean backward_tensor
#axiomsClean gradient_tensor
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelCorrect

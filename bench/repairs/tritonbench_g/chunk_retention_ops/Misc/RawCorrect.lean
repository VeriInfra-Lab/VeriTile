import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelForwardCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelOutputCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelBackwardCorrect
import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelGradientCorrect
/-! Complete independent mathematics for all four raw source kernels. These
exports take only scalar shapes/flags/program IDs and raw state or memory. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.RawCorrect
export KernelReal (forward_correct output_correct backward_correct gradient_correct)
#axiomsClean forward_correct
#axiomsClean output_correct
#axiomsClean backward_correct
#axiomsClean gradient_correct
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.RawCorrect

import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Misc.TensorBridge
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Correct
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.FPEquiv

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly
#check selectedKernel
#check source_correct
#check TensorBridge.correct
#check source_fp_equiv
#check TensorBridge.fp_equiv
#axiomsClean TensorBridge.correct
#axiomsClean TensorBridge.fp_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly

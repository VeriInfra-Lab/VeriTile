import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Contract
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Correct
import bench.repairs.tritonbench_g.chunk_retention.KernelOnly.Misc.NativeCorrect

namespace VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly.TensorBridge
/-- Actual Tensor input binding, all logical output reads on one shared physical
post-read, and simultaneous exterior/input-storage/region frames. -/
alias correct := KernelTensor.real_complete
/-- Source-derived actual Tensor logical observations and one shared post-read. -/
alias fp_equiv := KernelTensor.fp_complete
#axiomsClean correct
#axiomsClean fp_equiv
end VeriTile.Bench.Repairs.ChunkRetention.Whole001.KernelOnly.TensorBridge

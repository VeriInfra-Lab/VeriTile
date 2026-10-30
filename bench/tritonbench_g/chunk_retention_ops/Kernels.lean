import bench.tritonbench_g.chunk_retention_ops.Sources.Kernels
import bench.tritonbench_g.chunk_retention_ops.Sources.KernelPacket

/-! Standard entry aliases for the four reviewed complete quotations and their
selected kernel call/state. No source or execution definition is replaced. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Kernels
export Source (forwardH forwardO backwardH backwardQKV)
export KernelPacket (Call)
export KernelPacket.Call (kernel state)
#check forwardH
#check forwardO
#check backwardH
#check backwardQKV
#check kernel
#check state
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Kernels

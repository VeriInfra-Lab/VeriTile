import bench.repairs.tritonbench_g.chunk_retention_ops.Proofs.KernelTensorBacking

/-! Standard entry aliases for the reviewed kernel PID and actual Tensor
binding/separation contracts. This does not import the historical host contract. -/
namespace VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Contract
export KernelPacket (Call Footprint)
export KernelTensor (Role View Valid)
export KernelTensorBacking
  (Apart InputRole InputUsed InputIndex Inputs OutputUsed OutputIndex Outputs Buffers)
#check Valid
#check Buffers
#check Footprint
end VeriTile.Bench.Repairs.ChunkRetentionOps.Whole001.KernelOnly.Contract

import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.RealInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.NativeInterface
import bench.repairs.tritonbench_g.layer_norm_liger.Proofs.KernelStrides

-- Supporting-module audits remain independent of public readers.
-- The public/Tensor/retained-host audit entry is ../KernelAudit.lean.
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.LayerIO.forward_implements
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.LayerIO.backward_implements
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.LayerIO.forward_native
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.LayerIO.backward_native
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.SourceSafety.forward
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.SourceSafety.Backward.source_safe
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.KernelContract.backward_addresses
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.KernelContract.inactive_reference
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.KernelContract.singleton_rowStride
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.KernelContract.singleton_inactive_wrap
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.KernelContract.inactive_range

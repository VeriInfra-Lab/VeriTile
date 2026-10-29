import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.layer_norm_liger.Contract
import bench.repairs.tritonbench_g.layer_norm_liger.Correct
import bench.repairs.tritonbench_g.layer_norm_liger.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.Correctness.layer_norm_forward_kernel_real
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.Correctness.layer_norm_backward_kernel_real
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.Kernel.FPEquiv.forward_pair
#axiomsClean VeriTile.Bench.Repairs.LayerNormLiger.Kernel.FPEquiv.backward_pair

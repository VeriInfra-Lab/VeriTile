import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.fast_rms_layernorm.Contract
import bench.repairs.tritonbench_g.fast_rms_layernorm.Correct
import bench.repairs.tritonbench_g.fast_rms_layernorm.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.FastRmsLayernorm.Correct.kernel_forward
#axiomsClean VeriTile.Bench.Repairs.FastRmsLayernorm.Correct.kernel_backward
#axiomsClean VeriTile.Bench.Repairs.FastRmsLayernorm.FPEquiv.kernel_equivalent

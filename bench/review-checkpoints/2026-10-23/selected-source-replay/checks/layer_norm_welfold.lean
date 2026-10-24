import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.layer_norm_welfold.Contract
import bench.repairs.tritonbench_g.layer_norm_welfold.Correct
import bench.repairs.tritonbench_g.layer_norm_welfold.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.Correct.kernel_real
#axiomsClean VeriTile.Bench.Repairs.LayerNormWelfold.FPEquiv.row_offset_reused_equiv

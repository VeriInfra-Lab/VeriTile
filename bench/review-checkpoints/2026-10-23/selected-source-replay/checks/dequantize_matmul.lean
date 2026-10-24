import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.dequantize_matmul.Contract
import bench.repairs.tritonbench_g.dequantize_matmul.Correct
import bench.repairs.tritonbench_g.dequantize_matmul.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.DequantizeMatmul.Correct.kernel_correct
#axiomsClean VeriTile.Bench.Repairs.DequantizeMatmul.FPEquiv.kernel_equiv

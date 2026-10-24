import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Contract
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.Correct
import bench.repairs.tritonbench_g.triton_matmul.KernelOnly.FPEquiv

#axiomsClean VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.kernel_real
#axiomsClean VeriTile.Bench.TritonBenchG.TritonMatmul.Repair.KernelOnly.FPEquiv.grid_equiv

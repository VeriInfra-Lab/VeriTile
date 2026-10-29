import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.sin_kernel.Contract
import bench.repairs.tritonbench_g.sin_kernel.Correct
import bench.repairs.tritonbench_g.sin_kernel.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.SinKernel.Correct.sin_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.SinKernel.FPEquiv.index_commuted_equiv

import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.sin_computation.Contract
import bench.repairs.tritonbench_g.sin_computation.Correct
import bench.repairs.tritonbench_g.sin_computation.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.SinComputation.Contiguous.Correct.sin_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.SinComputation.Contiguous.FPEquiv.index_commuted_equiv

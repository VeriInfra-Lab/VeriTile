import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.cosine_compute.Contract
import bench.repairs.tritonbench_g.cosine_compute.Correct
import bench.repairs.tritonbench_g.cosine_compute.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.CosineCompute.Correct.cosine_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.CosineCompute.FPEquiv.index_commuted_equiv

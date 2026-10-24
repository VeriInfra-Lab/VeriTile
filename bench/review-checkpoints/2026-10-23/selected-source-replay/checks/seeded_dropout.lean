import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.seeded_dropout.Contract
import bench.repairs.tritonbench_g.seeded_dropout.Correct
import bench.repairs.tritonbench_g.seeded_dropout.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.SeededDropout.Correct.seeded_dropout_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.SeededDropout.FPEquiv.index_commuted_equiv

import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.cross_entropy2.Contract
import bench.repairs.tritonbench_g.cross_entropy2.Correct
import bench.repairs.tritonbench_g.cross_entropy2.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.CrossEntropy2.Correct.forward
#axiomsClean VeriTile.Bench.Repairs.CrossEntropy2.Correct.backward
#axiomsClean VeriTile.Bench.Repairs.CrossEntropy2.FPEquiv.forward_index_commuted_equiv
#axiomsClean VeriTile.Bench.Repairs.CrossEntropy2.FPEquiv.backward_index_commuted_equiv

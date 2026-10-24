import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.cross_entropy1.Contract
import bench.repairs.tritonbench_g.cross_entropy1.Correct
import bench.repairs.tritonbench_g.cross_entropy1.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.CrossEntropy1.Correct.forward_unsplit_kernel
#axiomsClean VeriTile.Bench.Repairs.CrossEntropy1.Correct.forward_split_kernel
#axiomsClean VeriTile.Bench.Repairs.CrossEntropy1.Correct.backward_kernel
#axiomsClean VeriTile.Bench.Repairs.CrossEntropy1.FPEquiv.forward_index_commuted_equiv
#axiomsClean VeriTile.Bench.Repairs.CrossEntropy1.FPEquiv.backward_index_commuted_equiv

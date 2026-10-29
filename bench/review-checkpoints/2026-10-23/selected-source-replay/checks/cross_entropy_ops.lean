import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.cross_entropy_ops.Contract
import bench.repairs.tritonbench_g.cross_entropy_ops.Correct
import bench.repairs.tritonbench_g.cross_entropy_ops.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.CrossEntropyOps.Correct.forward_unsplit
#axiomsClean VeriTile.Bench.Repairs.CrossEntropyOps.Correct.forward_split
#axiomsClean VeriTile.Bench.Repairs.CrossEntropyOps.Correct.backward
#axiomsClean VeriTile.Bench.Repairs.CrossEntropyOps.FPEquiv.forward_index_commuted_equiv
#axiomsClean VeriTile.Bench.Repairs.CrossEntropyOps.FPEquiv.backward_index_commuted_equiv

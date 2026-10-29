import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.log_softmax.Contract
import bench.repairs.tritonbench_g.log_softmax.Correct
import bench.repairs.tritonbench_g.log_softmax.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.LogSoftmax.Correct.forward_grid_correctness
#axiomsClean VeriTile.Bench.Repairs.LogSoftmax.Correct.backward_grid_correctness
#axiomsClean VeriTile.Bench.Repairs.LogSoftmax.FPEquiv.forward_index_equiv
#axiomsClean VeriTile.Bench.Repairs.LogSoftmax.FPEquiv.backward_index_equiv

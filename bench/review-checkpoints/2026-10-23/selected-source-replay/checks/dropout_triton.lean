import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.dropout_triton.Contract
import bench.repairs.tritonbench_g.dropout_triton.Correct
import bench.repairs.tritonbench_g.dropout_triton.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.DropoutTriton.Correct.dropout_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.DropoutTriton.FPEquiv.dropout_index_commuted_equiv

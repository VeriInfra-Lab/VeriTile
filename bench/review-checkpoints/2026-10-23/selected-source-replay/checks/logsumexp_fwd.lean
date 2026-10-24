import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.logsumexp_fwd.Contract
import bench.repairs.tritonbench_g.logsumexp_fwd.Correct
import bench.repairs.tritonbench_g.logsumexp_fwd.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.LogsumexpFwd.Correct.logsumexp_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.LogsumexpFwd.FPEquiv.source_equivalence

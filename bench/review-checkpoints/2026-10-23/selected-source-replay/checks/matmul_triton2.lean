import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Contract
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.Correct
import bench.repairs.tritonbench_g.matmul_triton2.KernelOnly.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.kernel_real
#axiomsClean VeriTile.Bench.Repairs.MatmulTriton2.KernelOnly.FPEquiv.source_pair_equiv

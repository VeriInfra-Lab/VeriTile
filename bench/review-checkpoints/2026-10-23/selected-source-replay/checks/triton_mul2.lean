import VeriTile.Meta.StatementAudit
import bench.repairs.tritonbench_g.triton_mul2.Contract
import bench.repairs.tritonbench_g.triton_mul2.Correct
import bench.repairs.tritonbench_g.triton_mul2.FPEquiv

#axiomsClean VeriTile.Bench.Repairs.TritonMul2.Correct.mul2_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.TritonMul2.Correct.mul2_inplace_kernel_correctness
#axiomsClean VeriTile.Bench.Repairs.TritonMul2.FPEquiv.mul2_commuted_equiv
#axiomsClean VeriTile.Bench.Repairs.TritonMul2.FPEquiv.mul2_inplace_commuted_equiv

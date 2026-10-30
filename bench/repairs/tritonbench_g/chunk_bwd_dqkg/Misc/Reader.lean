import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Misc.TensorBridge
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Source
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Contract
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.Correct
import bench.repairs.tritonbench_g.chunk_bwd_dqkg.FPEquiv

namespace VeriTile.Bench.Repairs.ChunkBwdDqkg.Reader
abbrev realApplication := @TensorBridge.kernel_real
abbrev nativeApplication := @TensorBridge.kernel_native
#axiomsClean realApplication
#axiomsClean nativeApplication
#auditStmt TensorBridge.kernel_real
#auditStmt TensorBridge.kernel_native
end VeriTile.Bench.Repairs.ChunkBwdDqkg.Reader

import bench.repairs.tritonbench_g.var_len_copy.Proofs.KernelRealStep
import bench.repairs.tritonbench_g.var_len_copy.Contract
import VeriTile.Triton.Memory.RegionalKernelIO
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.Repairs.VarLenCopy.PublicIO
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem sourceProjection : Source.originalKernel.eraseDType.toAlgorithm? =
    Except.ok Source.originalKernel.eraseDType.toAlgKernel := by
  simp [Source.originalKernel, Source.kernel,Source.source,Source.commutedSource, Source.blockSize,
    ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

end
end VeriTile.Bench.Repairs.VarLenCopy.PublicIO

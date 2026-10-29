import bench.tritonbench_g.matmul_persistent_triton.Proofs.Parameters
import VeriTile.Triton.Float.Erasure

namespace VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSource
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 16000

/-- The actual complete source algorithm after forgetting floating precision.
Signed schedule state, fused dot, all masks and resets remain in this kernel;
no K-loop limit or persistent output tile is replaced by a proof-side bound. -/
def kernel (p : Parameters) (v : Bool) : Kernel :=
  (p.source .fp16 v).eraseDType.toAlgKernel

/-- Both native storage arms have the same complete real algorithm projection.
This is a projection theorem, separate from typed numerical execution. -/
theorem projection (p : Parameters) (storage : TypedSource.Storage) (v : Bool) :
    (p.source storage v).eraseDType.toAlgorithm? = .ok (kernel p v) := by
  cases storage <;> cases v <;>
    simp only [kernel, Parameters.source, TypedSource.kernel,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeKernel.surfaceBody, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
      List.map_cons, List.map_nil, FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.MatmulPersistentTriton.RealSource

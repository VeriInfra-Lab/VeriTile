import bench.repairs.tritonbench_g.cosine_compute.Kernels
import bench.tritonbench_g.cosine_compute.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.CosineCompute.Projection
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.CosineCompute.RealProof
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 10000

/-- The source's explicit FP32 cast is retained. In the mathematical semantics
its conversion is an identity; this theorem is not an FP32 numerical law. -/
theorem kernel_execution (A O : RegionName) (n B : Nat) (s : BlockState) :
    exec (kernel A O n B).toAlgKernel s = exec (realKernel Bool.false A O n B).toAlgKernel s := by
  simp [kernel,realKernel,exec,ComputeKernel.toAlgKernel,ComputeKernel.surfaceBody,
    ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,stepStmts,stepStmt,evalOp.eq_def,
    Tile.bop,Tile.uop,Tile.cop,NumericDType.add,NumericDType.mul,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,Region.cast]

theorem kernel_flattenOk (A O : RegionName) (n B : Nat) : (kernel A O n B).toAlgKernel.FlattenOk := by
  simp [Kernel.FlattenOk,kernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeKernel.toAlgKernel,StmtList.FlattenOk,Stmt.FlattenOk,
    Op.FlattenOk.eq_def,Region.cast]

theorem kernel_traceSafe (A O : RegionName) (n B : Nat) (bounds : RegionBounds) (s : BlockState)
    (hi : ∀ j : Fin B,s.pid*B+j.val < n → s.pid*B+j.val < bounds A)
    (ho : ∀ j : Fin B,s.pid*B+j.val < n → s.pid*B+j.val < bounds O) :
    Kernel.TraceSafe bounds (kernel A O n B).toAlgKernel s := by
  simpa [Kernel.TraceSafe,kernel,realKernel,ComputeKernel.surfaceBody,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeKernel.toAlgKernel,Stmt.TraceSafeList,Stmt.TraceSafe,
    Op.SafeAt.eq_def,MaskOpt.SafeAt,stepStmt,evalOp.eq_def,Tile.bop,Tile.uop,Tile.cop,
    NumericDType.add,NumericDType.mul,ComparableDType.lt,WithBot.realCos,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,MemAccess.ActiveAddressSafe,
    memAccessActiveAddressSafe,MemAccess.SafeAt,MaskOpt.Active,BlockState.setReg,Region.cast] using
    cos_func_traceSafe A O n B bounds s hi ho

theorem original_projection (n : Nat) :
    (originalKernel n).toAlgorithm? = .ok (originalKernel n).toAlgKernel := by rfl

#axiomsClean kernel_execution
#axiomsClean kernel_flattenOk
#axiomsClean kernel_traceSafe
#axiomsClean original_projection
end
end VeriTile.Bench.Repairs.CosineCompute.Projection

import bench.tritonbench_g.dropout_triton.Proofs.Mathematical_Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DropoutTriton.Erasure
open Lean VeriTile.Triton TypedSource
set_option maxHeartbeats 3000000
set_option maxRecDepth 20000

/-- Mathematical precision projection, derived from the one shared source. -/
def source (keep : KeepStorage) (swapped : Bool) : sourceType :=
  fun x k y n B p => (TypedSource.source .fp32 keep swapped x k y n B p).eraseDType

theorem body (storage : Storage) (keep : KeepStorage) (swapped : Bool)
    (x mask out : RegionName) (n B : Nat) (p : ℝ) :
    (TypedSource.source storage keep swapped x mask out n B p).eraseDType.toAlgKernel.body =
      (source keep swapped x mask out n B p).toAlgKernel.body := by
  cases storage <;> cases keep <;> cases swapped
  all_goals simp only [TypedSource.source,source,VeriTile.Bench.Repairs.DropoutTriton.sourceFamily,
    ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DropoutTriton.Erasure

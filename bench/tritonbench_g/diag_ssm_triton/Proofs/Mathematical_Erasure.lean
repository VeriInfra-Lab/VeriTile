import bench.tritonbench_g.diag_ssm_triton.Sources.Mathematical_Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.DiagSsmTriton.Erasure
open VeriTile.Triton TypedSource
set_option maxHeartbeats 6000000
set_option maxRecDepth 20000

/-- Real projections of the shared repaired sources. No complete mathematical
kernel is copied into the proof layer. -/
def forward (swapped : Bool) : forwardSource :=
  fun s x lam y length batch dim B =>
    (TypedSource.forward .fp32 swapped s x lam y length batch dim B).eraseDType

def backward (swapped : Bool) : backwardSource :=
  fun s lam y gs gx glam gy length batch dim B =>
    (TypedSource.backward .fp32 swapped s lam y gs gx glam gy length batch dim B).eraseDType

def complexForward (swapped : Bool) : complexForwardSource :=
  fun s x y lam length batch dim B =>
    (TypedSource.complexForward swapped s x y lam length batch dim B).eraseDType

def complexBackward (swapped : Bool) : complexBackwardSource :=
  fun s lam y gs gx glam gy length batch dim B =>
    (TypedSource.complexBackward swapped s lam y gs gx glam gy length batch dim B).eraseDType

theorem forward_body (storage : Storage) (swapped : Bool) (s_ptr x_ptr lambda_ptr y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.forward storage swapped s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (forward swapped s_ptr x_ptr lambda_ptr y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases storage <;> cases swapped
  all_goals simp only [TypedSource.forward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.forwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.forwardIndexCommuted,forward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

theorem backward_body (storage : Storage) (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.backward storage swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (backward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases storage <;> cases swapped
  all_goals simp only [TypedSource.backward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.backwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.backwardIndexCommuted,backward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,FloatDType.toTileDType]
  all_goals rfl

theorem complexForward_body (swapped : Bool) (s_ptr x_ptr y_ptr lambda_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.complexForward swapped s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (complexForward swapped s_ptr x_ptr y_ptr lambda_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases swapped
  all_goals simp only [TypedSource.complexForward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexForwardIndexCommuted,complexForward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

theorem complexBackward_body (swapped : Bool) (s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr : RegionName) (length batch_size dim BLOCK_SIZE : Nat) :
    (TypedSource.complexBackward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).eraseDType.toAlgKernel.body =
      (complexBackward swapped s_ptr lambda_ptr y_ptr grad_s_ptr grad_x_ptr grad_lambda_ptr grad_y_ptr length batch_size dim BLOCK_SIZE).toAlgKernel.body := by
  cases swapped
  all_goals simp only [TypedSource.complexBackward,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardTyped,VeriTile.Bench.Repairs.DiagSsmTriton.Kernels.complexBackwardIndexCommuted,complexBackward,
    Bool.false_eq_true,↓reduceIte,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    ComputeDType.eraseDType,List.map_cons,List.map_nil,Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType,MemAccess.eraseDType,MaskOpt.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,Except.bind,bind]
  all_goals simp only [Op.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, FloatDType.toTileDType]
  all_goals rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.DiagSsmTriton.Erasure

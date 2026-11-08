import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualPrologue
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualRelocation
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualTail

/-! Assemble complete source projections, retaining all capture/return
instructions. Pointer-base relocation affects only the actual headerView. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualAssembly
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def headerView (p : Parameters) (BD : Nat) : List Stmt :=
  ActualSource.project ((TypedExecution.code (ActualSource.config p) BD).take 27)

theorem prefix_eq (p : Parameters) (BD : Nat) :
    headerView p BD=ActualSource.project (TypedExecution.coordinateCode (ActualSource.config p) BD) ++
      ActualSource.project (TypedExecution.pointerCode (ActualSource.config p) BD) ++
      ActualSource.project (TypedExecution.initializationCode (ActualSource.config p) BD) := by
  by_cases hd : p.stage=2
  all_goals simp only [headerView,ActualSource.project,TypedExecution.coordinateCode,
    TypedExecution.pointerCode,TypedExecution.initializationCode,TypedExecution.code,
    TypedExecution.source,ActualSource.config,TypedSource.repaired,originalKernel,
    hd,decide_true,decide_false,Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody,
    List.take_succ_cons,List.take_zero,List.drop_succ_cons,List.drop_zero]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp [Stmt.eraseDTypeList,List.append_assoc]

theorem assembled (p : Parameters) (BD : Nat) :
    Verified.code 128 64 BD p Bool.false=headerView p BD ++
      ActualCall.code p BD Bool.false ++ ActualCall.code p BD Bool.true ++ ActualTail.code p BD := by
  rw [ActualSource.assembled,prefix_eq]
  simp only [ActualSource.chunks,ActualCall.code,ActualSetup.code,ActualTail.code,
    List.map_cons,List.map_nil,List.flatten_cons,List.flatten_nil,List.append_nil,List.append_assoc]

theorem flat (A : FlatAlloc) (p : Parameters) (BD : Nat) :
    A.flattenStmts (Verified.code 128 64 BD p Bool.false)=ActualPrologue.code A p BD ++
      ActualCall.code (Relocation.parameters p A.flat) BD Bool.false ++
      ActualCall.code (Relocation.parameters p A.flat) BD Bool.true ++ ActualTail.code p BD := by
  rw [assembled]
  simp only [ActualRelocation.append,ActualRelocation.call_flat,ActualTail.flat]
  rfl

#axiomsClean flat
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualAssembly

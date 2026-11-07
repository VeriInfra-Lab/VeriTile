import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualCall
import bench.tritonbench_g.attn_fwd_triton.Proofs.Relocation
import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatCongruence

/-! Helper calls contain only captured pointers; changing backing-region
names leaves their complete source projection unchanged. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualRelocation
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem setup_flat (A : FlatAlloc) (p : Parameters) (BD : Nat) (second : Bool) :
    A.flattenStmts (ActualSetup.code p BD second)=
      ActualSetup.code (Relocation.parameters p A.flat) BD second := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [ActualSetup.code,ActualSource.project,TypedExecution.setupCode,TypedExecution.setup,
    TypedExecution.preloop,TypedExecution.parts,TypedExecution.code,TypedExecution.source,
    ActualSource.config,Relocation.parameters,TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals simp [Source.intRange?,TypedExecution.workName]
  all_goals try simp only [List.drop_succ_cons,List.drop_zero]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals simp [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask]

theorem range_flat (A : FlatAlloc) (p : Parameters) (BD : Nat) (second : Bool) :
    A.flattenStmt (ActualRanges.statement p BD second)=
      ActualRanges.statement (Relocation.parameters p A.flat) BD second := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [ActualRanges.statement,HelperRegisters.body,HelperRegisters.parts,
    Verified.source,Relocation.parameters,TypedSource.repaired,originalKernel,
    hd,decide_true,decide_false,Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals simp [Source.intRange?,HelperRegisters.workName,TypedExecution.workName]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals simp [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask]
  all_goals repeat' apply And.intro
  all_goals erw [FlatAlloc.flattenOp]; simp only [FlatAlloc.flattenOp]

theorem return_flat (A : FlatAlloc) (p : Parameters) (BD : Nat) (second : Bool) :
    A.flattenStmts (ActualReturn.code p BD second)=
      ActualReturn.code (Relocation.parameters p A.flat) BD second := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [ActualReturn.code,ActualSource.project,TypedExecution.finish,TypedExecution.parts,
    TypedExecution.code,TypedExecution.source,ActualSource.config,Relocation.parameters,
    TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals simp [Source.intRange?,TypedExecution.workName]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals simp [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask]

theorem append (A : FlatAlloc) (left right : List Stmt) :
    A.flattenStmts (left++right)=A.flattenStmts left++A.flattenStmts right := by
  induction left with
  | nil=>rfl
  | cons head tail ih=>simp only [List.cons_append,FlatAlloc.flattenStmts,ih]

theorem call_flat (A : FlatAlloc) (p : Parameters) (BD : Nat) (second : Bool) :
    A.flattenStmts (ActualCall.code p BD second)=
      ActualCall.code (Relocation.parameters p A.flat) BD second := by
  simp only [ActualCall.code,append,ActualRanges.source,FlatAlloc.flattenStmts,
    setup_flat,range_flat,return_flat]

#axiomsClean call_flat

end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualRelocation

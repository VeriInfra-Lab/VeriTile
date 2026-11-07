import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualSource
import bench.tritonbench_g.attn_fwd_triton.Proofs.Output
import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatCongruence

/-! The output normalization and dtype-erased bf16 store are projected
from the complete pasted caller, rather than appended to a separate kernel. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualTail
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def code (p : Parameters) (BD : Nat) : List Stmt :=
  ActualSource.project (TypedExecution.tail (ActualSource.config p) BD Bool.false)

theorem source (p : Parameters) (BD : Nat) : code p BD=epilogue 128 BD p.headDim Bool.false := by
  by_cases hd : p.stage=2
  all_goals simp only [code,ActualSource.project,TypedExecution.tail,TypedExecution.parts,
    TypedExecution.code,TypedExecution.source,ActualSource.config,
    TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals simp [Source.intRange?,TypedExecution.workName]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals simp [epilogue,RealSource.normalization,queryMask]

theorem flat (A : FlatAlloc) (p : Parameters) (BD : Nat) :
    A.flattenStmts (code p BD)=code p BD := by
  rw [source]
  simp [epilogue,RealSource.normalization,queryMask,
    FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask]
  all_goals repeat' apply And.intro
  all_goals erw [FlatAlloc.flattenOp]; simp only [FlatAlloc.flattenOp]

#axiomsClean source
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualTail

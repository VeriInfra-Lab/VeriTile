import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualSource
import bench.tritonbench_g.attn_fwd_triton.Proofs.HelperLoops
import bench.tritonbench_g.attn_fwd_triton.Proofs.Coordinates
import bench.tritonbench_g.attn_fwd_triton.Proofs.Relocation
import bench.tritonbench_g.attn_fwd_triton.Proofs.FlatPrologue

/-! Execute the actual caller prefix, including all casts, typed zero
construction, strided pointers and masked query/scale loads. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualPrologue
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
open VeriTile.Bench.TritonBenchG.AttentionScore.RealFrame (Code)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles (finite)
noncomputable section
set_option maxRecDepth 12000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

def code (A : FlatAlloc) (p : Parameters) (BD : Nat) : List Stmt :=
  A.flattenStmts (ActualSource.project ((TypedExecution.code (ActualSource.config p) BD).take 27))

def writes : List RegName := Coordinates.writes ++
  ["offs_m","offs_n","offs_k","Q_ptrs","Q_scale_ptr","K_ptrs","K_scale_ptr","V_ptrs","O_block_ptr",
   "m_i","l_i","acc","q","q_scale"]

theorem footprint (A : FlatAlloc) (p : Parameters) (BD : Nat) : Code writes (code A p BD) := by
  by_cases hd : p.stage=2
  all_goals simp only [code,ActualSource.project,TypedExecution.code,TypedExecution.source,
    ActualSource.config,TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody,
    List.take_succ_cons,List.take_zero]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,FloatDType.eraseFloat,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals simp only [FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask]
  all_goals repeat' first | exact Code.nil | apply Code.assign | (simp [writes,Coordinates.writes])

theorem code_eq (A : FlatAlloc) (p : Parameters) (BD : Nat) :
    code A p BD=Coordinates.code p 128 64 ++
      A.flattenStmts (Pointers.code p 128 64 BD) ++ Initialization.code 128 BD p.headDim := by
  by_cases hd : p.stage=2
  all_goals simp only [code,ActualSource.project,TypedExecution.code,TypedExecution.source,
    ActualSource.config,TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody,
    List.take_succ_cons,List.take_zero]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,FloatDType.eraseFloat,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals norm_num [Coordinates.code,Coordinates.runtimeInteger,Coordinates.ceilExpr,
    Pointers.code,Pointers.rowExpr,Pointers.rowOffsets,Pointers.keyExpr,
    Pointers.queryScaleExpr,Pointers.keyScaleExpr,Initialization.code,RealInit.code,
    queryMask,wide,FlatAlloc.flattenStmts,FlatAlloc.flattenStmt,FlatAlloc.flattenOp,
    FlatAlloc.flattenAccess,FlatAlloc.flattenMask]

theorem executes (A : FlatAlloc) (s : BlockState) (p : Parameters) (BD : Nat) :
    let c := Relocation.config A (Coordinates.configuration s p)
    ∃ t, stepStmts (code A p BD) s=some t ∧
      Iteration.Fixed s t c 128 64 BD ∧
      Iteration.Carries t (RealLoopFormula.initial 128 BD) ∧
      FirstSetup.Caller t c 64 BD ∧
      t.regs .ptr [128,BD] "O_block_ptr"=some (outputPointer c 128 BD) ∧ Frame writes s t := by
  rw [code_eq]
  simpa only [writes,FlatPrologue.writes,Prologue.writes,Coordinates.writes,
    Pointers.writes,Initialization.writes,RealInit.writes,List.cons_append,List.nil_append] using
    FlatPrologue.executes A s p 128 64 BD

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualPrologue

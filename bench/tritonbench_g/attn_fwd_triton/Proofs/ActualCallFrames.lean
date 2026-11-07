import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualSetup
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualReturn

/-! Finite register footprints of complete source call setup and returns. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualCallFrames
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionScore.RealFrame (Code)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem setup (p : Parameters) (BD : Nat) (second : Bool) :
    Code ActualFrames.workspace (ActualSetup.code p BD second) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [ActualSetup.code,ActualSource.project,TypedExecution.setupCode,TypedExecution.setup,
    TypedExecution.preloop,TypedExecution.parts,TypedExecution.code,TypedExecution.source,
    ActualSource.config,TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals simp [Source.intRange?,TypedExecution.workName]
  all_goals try simp only [List.drop_succ_cons,List.drop_zero]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals repeat' first | exact Code.nil | apply Code.assign | decide

theorem result (p : Parameters) (BD : Nat) (second : Bool) :
    Code (ActualReturn.writes second) (ActualReturn.code p BD second) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [ActualReturn.code,ActualSource.project,TypedExecution.finish,TypedExecution.parts,
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
  all_goals repeat' first | exact Code.nil | apply Code.assign | decide

theorem setup_frame {p : Parameters} {BD : Nat} {second : Bool} {s t : BlockState}
    (ran : stepStmts (ActualSetup.code p BD second) s=some t) :
    Frame ActualFrames.workspace s t := by
  have f := (setup p BD second).frame ran
  exact ⟨f.mem,f.pids,f.undef,f.regs⟩

theorem return_frame {p : Parameters} {BD : Nat} {second : Bool} {s t : BlockState}
    (ran : stepStmts (ActualReturn.code p BD second) s=some t) :
    Frame (ActualReturn.writes second) s t := by
  have f := (result p BD second).frame ran
  exact ⟨f.mem,f.pids,f.undef,f.regs⟩

#axiomsClean setup_frame
#axiomsClean return_frame
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualCallFrames

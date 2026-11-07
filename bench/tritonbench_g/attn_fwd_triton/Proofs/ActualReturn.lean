import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualSource
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualFrames

/-! Execute all six tuple-result assignments. The second Python call writes
its final maximum to `_`, preserving the caller's existing maximum. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualReturn
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
open RegisterViews HelperRegisters
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def code (p : Parameters) (BD : Nat) (second : Bool) : List Stmt :=
  ActualSource.project (TypedExecution.finish (ActualSource.config p) BD second)

def returned {BM BD : Nat} (second : Bool) (caller result : Carry BM BD) : Carry BM BD :=
  ⟨if second then caller.center else result.center,result.denominator,result.numerator⟩

def writes (second : Bool) : List RegName :=
  ActualFrames.workspace ++ ["acc","l_i",if second then "__veritile_python_underscore" else "m_i"]

theorem executes (s : BlockState) (p : Parameters) (BD : Nat) (second : Bool)
    (caller result : Carry 128 BD)
    (helper : Iteration.Carries (pull (names second) s) result)
    (old : s.regs .real [128] "m_i"=some (⟨fun i=>caller.center i.1⟩ : Tile .real [128])) :
    ∃ t, stepStmts (code p BD second) s=some t ∧
      Iteration.Carries t (returned second caller result) := by
  have hm : s.regs .real [128] (workName second "m_i")=
      some (⟨fun i=>result.center i.1⟩ : Tile .real [128]) := by
    simpa [pull] using helper.center
  have hl : s.regs .real [128] (workName second "l_i")=
      some (VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles.finite (fun i : TileIndex [128]=>result.denominator i.1)) := by
    simpa [pull] using helper.denominator
  have ha : s.regs .real [128,BD] (workName second "acc")=
      some (VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealTiles.finite result.numerator) := by
    simpa [pull] using helper.numerator
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [code,ActualSource.project,TypedExecution.finish,TypedExecution.parts,
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
  all_goals simp [stepStmts,stepStmt,evalOp.eq_def,Option.bind,BlockState.setReg,
    Iteration.Carries,RealIteration.Carries,workName,TypedExecution.workName] at hm hl ha ⊢
  all_goals simp [hm,hl,ha,old,returned]
  all_goals constructor <;> simp [old]

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualReturn

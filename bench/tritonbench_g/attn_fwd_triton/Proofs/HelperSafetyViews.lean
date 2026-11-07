import bench.tritonbench_g.attn_fwd_triton.Proofs.RegisterSafetyViews
import bench.tritonbench_g.attn_fwd_triton.Proofs.HelperRegisters

/-! Address-safety correspondence for every actual helper instruction. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HelperSafetyViews
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
open RegisterViews HelperRegisters
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 6000000
set_option linter.unusedSimpArgs false

private theorem assign_named (permutation : RegName ≃ RegName)
    (name actualName : RegName) {d : TileDType} {shape : TileShape}
    (view actual : Op d shape) (named : permutation name=actualName)
    (expression : ∀ s,evalOp actual s=evalOp view (pull permutation s))
    (access : ∀ bounds s,actual.SafeAt bounds s ↔ view.SafeAt bounds (pull permutation s))
    (before after : List Stmt) (rest : RegisterSafetyViews.Corresponds permutation before after) :
    RegisterSafetyViews.Corresponds permutation (.assign d shape name view::before)
      (.assign d shape actualName actual::after) := by
  rw [←named]
  exact RegisterSafetyViews.assign permutation name view actual expression access before after rest

theorem iteration (BM BN BD : Nat) (p : Parameters) (second : Bool) :
    RegisterSafetyViews.Corresponds (names second)
      (sourceIteration BM BN BD p.headDim second (decide (p.stage=2)))
      (body BM BN BD p second) := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [body,parts,source,TypedSource.repaired,originalKernel,
    hd,decide_true,decide_false,workName,AttnFwdTriton.TypedExecution.workName,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody]
  all_goals simp [Source.intRange?,workName,AttnFwdTriton.TypedExecution.workName]
  all_goals try dsimp only [ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals try simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,FloatDType.eraseFloat,NumericDType.eraseDType,
    ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals simp only [sourceIteration,flatIteration,loadPrefix,guard,suffix,
    valueLoad,advance,RealBody.score,RealBody.scoreExpr,RealBody.center,
    RealBody.centerExpr,RealBody.probability,RealBody.rescale,RealBody.accumulate,
    Bool.false_or,Bool.true_or,Bool.false_eq_true,Bool.true_eq,↓reduceIte,
    List.nil_append,List.cons_append]
  all_goals repeat' first | apply assign_named | exact RegisterSafetyViews.nil _
  all_goals try simp [
    workName,AttnFwdTriton.TypedExecution.workName,keyName,scaleName,valueName,maskName,
    RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName,RealPreparation.maskName]
  all_goals intros
  all_goals simp [Op.SafeAt.eq_def,Op.MemorySafe,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,Op.PointerAddressesSafeOn,MaskOpt.Active,evalOp.eq_def,pull,
    workName,AttnFwdTriton.TypedExecution.workName,keyName,scaleName,valueName,maskName,
    RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName,RealPreparation.maskName,
    maskExpr,tail,keyMask,valueMask,wide,BlockState.readMem]
  all_goals norm_num

#axiomsClean iteration
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.HelperSafetyViews

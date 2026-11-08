import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualCallFrames

/-! The signed source range, including its source-derived hygienic counter,
body and explicit int64 bounds. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualRanges
open VeriTile.Triton
open RegisterViews HelperRegisters
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def statement (p : Parameters) (BD : Nat) (second : Bool) : Stmt :=
  .forRangeInt (workName second "start_n") (.ref .int [] (workName second "lo"))
    (.ref .int [] (workName second "hi")) (.constNat 64)
    (HelperRegisters.body 128 64 BD p second)

theorem source (p : Parameters) (BD : Nat) (second : Bool) :
    ActualSource.project (TypedExecution.loopCode (ActualSource.config p) BD second)=
      statement p BD second::ActualReturn.code p BD second := by
  cases second <;> by_cases hd : p.stage=2
  all_goals simp only [ActualSource.project,TypedExecution.loopCode,TypedExecution.body,
    TypedExecution.finish,TypedExecution.parts,TypedExecution.code,TypedExecution.source,
    ActualSource.config,TypedSource.repaired,originalKernel,hd,decide_true,decide_false,
    Bool.false_eq_true,Bool.true_eq,↓reduceIte,ComputeKernel.surfaceBody,
    ActualReturn.code,statement,HelperRegisters.body,HelperRegisters.parts,Verified.source,
    TypedExecution.lowerOp,TypedExecution.lowerName,TypedExecution.upperName]
  all_goals simp [Source.intRange?,workName,TypedExecution.workName]
  all_goals dsimp only [ComputeKernel.toAlgKernel,ComputeKernel.toAlgorithm?,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?,ComputeDType.eraseDType,Except.bind,bind]
  all_goals simp only [Stmt.eraseDTypeList,Stmt.eraseDType,Op.eraseDType.eq_def,
    MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,VeriTile.Triton.eraseDType,
    FloatDType.toTileDType,NumericDType.eraseDType,ComparableDType.eraseDType,IntegralDType.eraseDType]
  all_goals rfl

theorem executes {p : Parameters} {BD : Nat} {second : Bool} {s t : BlockState} {lower upper : Nat}
    (lo : s.regs .int [] (workName second "lo")=some (Tile.scalar (lower : Int)))
    (hi : s.regs .int [] (workName second "hi")=some (Tile.scalar (upper : Int)))
    (loop : stepForRangeIntAux (names second "start_n") lower upper 64
      (HelperRegisters.body 128 64 BD p second) s=some t) :
    stepStmt (statement p BD second) s=some t := by
  simpa only [statement,names_start_n,stepStmt,evalOp_ref,evalOp_constNat,
    lo,hi,Option.bind_some,Tile.scalar,bind] using loop

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualRanges

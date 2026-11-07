import bench.tritonbench_g.attn_fwd_triton.Proofs.HelperSafetyViews
import bench.tritonbench_g.attn_fwd_triton.Proofs.SafetyLoop
import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualMiddle

/-! Trace safety of the actual signed helper range, including captured
registers and all source argument/return assignments. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualHelperSafety
open VeriTile.Triton Memory RegisterViews HelperRegisters
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
noncomputable section
set_option maxRecDepth 32000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

theorem range_safe (bounds : RegionBounds) (mem s : BlockState) (c : Config)
    (BD lower upper : Nat) (second : Bool) (seed : Carry 128 BD)
    (windows : SafetyLoop.Bounds bounds c 64 BD lower upper)
    (ready : Loop.Registers mem (pull (names second) s) c 128 64 BD lower second seed) :
    Stmt.forRangeIntTraceSafe bounds (workName second "start_n") lower upper 64
      (HelperRegisters.body 128 64 BD c.p second) s := by
  have old := SafetyLoop.loop_safe bounds mem (pull (names second) s) c lower upper second
    (decide (c.p.stage=2)) seed (by decide) windows ready
  have safe := (RegisterSafetyViews.signed_range (names second) "start_n"
    (sourceIteration 128 64 BD c.p.headDim second (decide (c.p.stage=2)))
    (HelperRegisters.body 128 64 BD c.p second)
    (iteration_corresponds 128 64 BD c.p second)
    (HelperSafetyViews.iteration 128 64 BD c.p second) bounds lower upper 64 s).mpr old
  simpa only [names_start_n] using safe

theorem statement_safe (bounds : RegionBounds) (mem s : BlockState) (c : Config)
    (BD lower upper : Nat) (second : Bool) (seed : Carry 128 BD)
    (windows : SafetyLoop.Bounds bounds c 64 BD lower upper)
    (ready : Loop.Registers mem (pull (names second) s) c 128 64 BD lower second seed)
    (lo : s.regs .int [] (workName second "lo")=some (Tile.scalar (lower : Int)))
    (hi : s.regs .int [] (workName second "hi")=some (Tile.scalar (upper : Int))) :
    Stmt.TraceSafe bounds (ActualRanges.statement c.p BD second) s := by
  have safe := range_safe bounds mem s c BD lower upper second seed windows ready
  simpa only [ActualRanges.statement,Stmt.TraceSafe,Op.SafeAt.eq_def,
    evalOp_ref,evalOp_constNat,lo,hi,bind,Option.bind,Tile.scalar,true_and] using safe

theorem setup_safe (bounds : RegionBounds) (s : BlockState) (p : Parameters)
    (BD : Nat) (second : Bool) :
    Stmt.TraceSafeList bounds (ActualSetup.code p BD second) s := by
  apply list_safe
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
  all_goals simp [Op.SafeAt.eq_def]

theorem return_safe (bounds : RegionBounds) (s : BlockState) (p : Parameters)
    (BD : Nat) (second : Bool) :
    Stmt.TraceSafeList bounds (ActualReturn.code p BD second) s := by
  apply list_safe
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
  all_goals simp [Op.SafeAt.eq_def]

theorem call_safe (bounds : RegionBounds) (mem s : BlockState) (c : Config)
    (BD : Nat) (second : Bool) (seed : Carry 128 BD)
    (windows : SafetyLoop.Bounds bounds c 64 BD (ActualSetup.lower c second) (ActualSetup.upper c second))
    (fixed : Iteration.Fixed mem s c 128 64 BD)
    (carry : Iteration.Carries s seed) (caller : FirstSetup.Caller s c 64 BD) :
    Stmt.TraceSafeList bounds (ActualCall.code c.p BD second) s := by
  obtain ⟨a,ran,ready,lo,hi⟩ := ActualSetup.executes mem s c BD second seed fixed carry caller
  rw [ActualCall.code,ActualRanges.source]
  apply append_of_run (setup_safe bounds s c.p BD second) ran
  apply Stmt.TraceSafeList.cons_intro
    (statement_safe bounds mem a c BD _ _ second seed windows ready lo hi)
  intro t _
  exact return_safe bounds t c.p BD second

#axiomsClean call_safe
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualHelperSafety

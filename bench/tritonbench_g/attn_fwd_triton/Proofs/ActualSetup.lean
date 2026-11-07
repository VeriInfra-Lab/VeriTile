import bench.tritonbench_g.attn_fwd_triton.Proofs.ActualSource
import bench.tritonbench_g.attn_fwd_triton.Proofs.HelperLoops
import bench.tritonbench_g.attn_fwd_triton.Proofs.FirstSetup

/-! Execute the complete source helper captures and range initialization. -/
namespace VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualSetup
open VeriTile.Triton Memory
open VeriTile.Bench.TritonBenchG.AttnFwdTritonSupport
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealLoopFormula (Carry)
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealScores (Frame)
open RegisterViews HelperRegisters
noncomputable section
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000
set_option linter.unusedSimpArgs false

def code (p : Parameters) (BD : Nat) (second : Bool) : List Stmt :=
  ActualSource.project (TypedExecution.setupCode (ActualSource.config p) BD second)

def lower (c : Config) (second : Bool) : Nat :=
  if second || decide (c.p.stage=2) then c.startM*128 else 0

def upper (c : Config) (second : Bool) : Nat :=
  if second || decide (c.p.stage=2) then min ((c.startM+1)*128) c.p.context else c.startM*128

private theorem natural_product (a b : Nat) :
    ((a : Int)*((b : Int)*128)).toNat=a*(b*128) := by
  change ((a : Int)*((b : Int)*((128 : Nat) : Int))).toNat=_
  rw [←Int.natCast_mul,←Int.natCast_mul,Int.toNat_natCast]

private theorem choose_min (a b : Int) : (if a<b then a else b)=min a b := by
  by_cases h : a<b
  · simp only [h,↓reduceIte,min_eq_left (le_of_lt h)]
  · simp only [h,↓reduceIte,min_eq_right (le_of_not_gt h)]

theorem executes (mem s : BlockState) (c : Config) (BD : Nat) (second : Bool)
    (seed : Carry 128 BD)
    (fixed : Iteration.Fixed mem s c 128 64 BD)
    (carry : Iteration.Carries s seed)
    (caller : FirstSetup.Caller s c 64 BD) :
    ∃ t, stepStmts (code c.p BD second) s=some t ∧
      Loop.Registers mem (pull (names second) t) c 128 64 BD (lower c second) second seed ∧
      t.regs .int [] (workName second "lo")=some (Tile.scalar (lower c second)) ∧
      t.regs .int [] (workName second "hi")=some (Tile.scalar (upper c second)) := by
  rcases fixed with ⟨hmemory,hq,hqs,hrows,hkeys,hctx,hkr,hvr⟩
  rcases carry with ⟨hmi,hli,hacc⟩
  rcases caller with ⟨hstart,hkp,hksp,hvp⟩
  cases second <;> by_cases hd : c.p.stage=2
  all_goals simp only [code,ActualSource.project,TypedExecution.setupCode,TypedExecution.setup,
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
  all_goals simp [stepStmts,stepStmt,evalOp.eq_def,Option.bind,BlockState.setReg,
    hq,hqs,hrows,hkeys,hctx,hkr,hvr,hmi,hli,hacc,hstart,hkp,hksp,hvp,
    Loop.Registers,Iteration.Fixed,Iteration.Carries,RealIteration.Carries,Preparation.Pointers,
    pull,lower,upper,hd,workName,keyName,scaleName,valueName]

  all_goals repeat' constructor
  all_goals try simp [pull,workName,TypedExecution.workName,
    keyName,scaleName,valueName,RealMemory.keyName,RealMemory.scaleName,RealMemory.valueName,
    hmemory,NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,
    -Nat.cast_add,-Nat.cast_mul,←Int.natCast_add,←Int.natCast_mul,
    Int.natCast_ediv,Int.toNat_natCast]
  all_goals apply Tile.ext
  all_goals intro i
  all_goals try simp [Tile.bop_data,Tile.cop_data,Tile.select_data,Tile.scalar,
    Tile.ptrAdd_data,keyPointer,keyScalePointer,valuePointer,
    RealPointers.rowPointer,Memory.rows,Tile.vec,NumericDType.add,NumericDType.mul,
    NumericDType.sub,ComparableDType.lt,TileShape.dropInsertedIndex,
    -Nat.cast_add,-Nat.cast_mul,←Int.natCast_add,←Int.natCast_mul,
    Int.natCast_ediv,Int.toNat_natCast,-Tile.scalar_eta]
  all_goals first
    | exact choose_min _ _
    | { rw [natural_product]; ring }

#axiomsClean executes
end
end VeriTile.Bench.Repairs.AttnFwdTriton.Verified.ActualSetup

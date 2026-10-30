import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteLoop
import bench.tritonbench_g.parallel_attention.Sources.Mathematical_FPRange

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
open VeriTile.Bench.TritonBenchG.ParallelAttention.FPStages (foldRange)
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

theorem rangeExecution {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (base current : State α)
    (t stop : Nat) (bound : t ≤ stop) (old : TileIndex [c.BM,c.BN] → Int)
    (ctx : Context s c base t current) (acc : HasAcc c old current) :
    ∃ final, range M q s.M s.N "k" t stop 1 (body s c bias) current=some final ∧
      Context s c base stop final ∧
      HasAcc c (foldRange (update q s c base) t stop 1 old) final := by
  by_cases active : t<stop
  · let next:=bodyExecution M q s c bias base current t old ctx acc
    have nextContext:=bodyContext M q s c bias base current t old ctx acc
    obtain ⟨final,runFinal,contextFinal,accFinal⟩:=rangeExecution M q s c bias base next.val
      (t+1) stop (by omega) (update q s c base t old) nextContext.1 nextContext.2
    refine ⟨final,?_,contextFinal,?_⟩
    · rw [range]
      simp only [show 0<1 ∧ t<stop from ⟨by decide,active⟩,↓reduceIte,next.property,Option.bind_some]
      exact runFinal
    · rw [foldRange]
      simpa only [show (1:Nat)≠0 by decide,active,if_false,if_true] using accFinal
  · have same : t=stop := by omega
    subst stop
    refine ⟨current,?_,ctx,?_⟩
    · simp only [range,lt_self_iff_false,and_false,↓reduceIte]
    · rw [foldRange]
      simpa only [show (1:Nat)≠0 by decide,lt_self_iff_false,if_false] using acc
termination_by stop-t
decreasing_by omega

def accumulation {α : Type} [Inhabited α] (q : IntegerModel I F H)
    (s : Shape) (c : Config) (base : State α) : TileIndex [c.BM,c.BN] → Int :=
  foldRange (update q s c base) 0 (steps s c) 1 (fun _=>0)

def offset {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α)
    (i : TileIndex [c.BM,c.BN]) : Nat :=
  (register base .int [c.BM] "rm" (i.1,PUnit.unit)*(s.N:Int)+
    register base .int [c.BN] "rn" (i.2.1,PUnit.unit)).toNat
def active {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α)
    (i : TileIndex [c.BM,c.BN]) : Prop :=
  register base .int [c.BM] "rm" (i.1,PUnit.unit)<(s.M:Int) ∧
    register base .int [c.BN] "rn" (i.2.1,PUnit.unit)<(s.N:Int)
instance {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α) :
    DecidablePred (active s c base) := fun _=>inferInstanceAs (Decidable (_∧_))

def epilogueExecution {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (base current : State α)
    (divisor : α) (hasDivisor : base.regs .fp32 [] "divfactor"=some (fun _=>divisor))
    (atomic : α→α→α) (hasAtomic : M.atomicAdd .fp16=some atomic)
    (ctx : Context s c base (steps s c) current)
    (acc : HasAcc c (accumulation q s c base) current) :
    {final : State α // run M q s.M s.N (epilogue s c bias) current=some final} := by
  simp only [HasAcc] at acc
  by_cases split : c.S=1
  all_goals
    cases bias
    all_goals
      by_cases first : register base .int [] "pid_z" PUnit.unit=0
      all_goals
        refine ⟨?_,?_⟩
        swap
        simp [epilogue,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
          TypedSource.runtimeScaled,ComputeKernel.surfaceBody,run,step,arithmeticStep,addExpression,
          FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
          numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
          FloatDType.numericDType,FloatDType.toTileDType,VeriTile.Triton.Region.cast,
          Option.bind,integralFloorDiv,integralMod,ctx.memory,ctx.pm,ctx.pn,ctx.pz,
          ctx.rm,ctx.rn,ctx.wx,ctx.xx,ctx.divisor,hasDivisor,acc,split,first,
          FP.Structural.atomicAdd,atomicAddition,hasAtomic]
        rfl

#axiomsClean rangeExecution
#axiomsClean epilogueExecution
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote

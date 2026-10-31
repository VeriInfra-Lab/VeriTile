import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteStages

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def register {α : Type} [Inhabited α] (s : State α) (d : TileDType)
    (shape : TileShape) (name : RegName) : Values α d shape :=
  (s.regs d shape name).getD (fun _=>defaultValue d)

def ap {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α)
    (t : Nat) (i : TileIndex [c.BM,c.BK]) : RegionName × Nat :=
  let p:=register base .ptr [c.BM,c.BK] "A" i
  (p.1,p.2+t*(c.BK*c.S))
def bp {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α)
    (t : Nat) (i : TileIndex [c.BK,c.BN]) : RegionName × Nat :=
  let p:=register base .ptr [c.BK,c.BN] "B" i
  (p.1,p.2+t*(c.BK*c.S*s.N))
@[simp] theorem ap_zero {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α) :
    ap s c base 0=register base .ptr [c.BM,c.BK] "A" := by
  funext i;simp [ap]
@[simp] theorem bp_zero {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α) :
    bp s c base 0=register base .ptr [c.BK,c.BN] "B" := by
  funext i;simp [bp]
def av {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α)
    (t : Nat) (i : TileIndex [c.BM,c.BK]) : Int :=
  if s.K%(c.BK*c.S)==0 then (base.mem (ap s c base t i).1 (ap s c base t i).2).read .int
  else if register base .int [c.BK] "rk" (i.2.1,PUnit.unit)<(s.K-t*(c.BK*c.S) : Nat) then
    (base.mem (ap s c base t i).1 (ap s c base t i).2).read .int else 0
def bv {α : Type} [Inhabited α] (s : Shape) (c : Config) (base : State α)
    (t : Nat) (i : TileIndex [c.BK,c.BN]) : Int :=
  if s.K%(c.BK*c.S)==0 then (base.mem (bp s c base t i).1 (bp s c base t i).2).read .int
  else if register base .int [c.BK] "rk" (i.1,PUnit.unit)<(s.K-t*(c.BK*c.S) : Nat) then
    (base.mem (bp s c base t i).1 (bp s c base t i).2).read .int else 0

def update {α : Type} [Inhabited α] (q : IntegerModel I F H)
    (s : Shape) (c : Config) (base : State α) (t : Nat)
    (old : TileIndex [c.BM,c.BN] → Int) : TileIndex [c.BM,c.BN] → Int :=
  dotAdd q s.M s.N (base.setReg "k" .nat [] (fun _=>t)) old (av s c base t) (bv s c base t)

structure Context {α : Type} [Inhabited α] (s : Shape) (c : Config)
    (base : State α) (t : Nat) (current : State α) : Prop where
  memory : current.mem=base.mem
  pids : current.pids=base.pids
  pm : current.regs .int [] "pid_m"=some (register base .int [] "pid_m")
  pn : current.regs .int [] "pid_n"=some (register base .int [] "pid_n")
  pz : current.regs .int [] "pid_z"=some (register base .int [] "pid_z")
  rm : current.regs .int [c.BM] "rm"=some (register base .int [c.BM] "rm")
  rn : current.regs .int [c.BN] "rn"=some (register base .int [c.BN] "rn")
  rk : current.regs .int [c.BK] "rk"=some (register base .int [c.BK] "rk")
  wx : current.regs .real [1,c.BN] "w_factor"=some (register base .real [1,c.BN] "w_factor")
  xx : current.regs .real [c.BM,1] "x_factor"=some (register base .real [c.BM,1] "x_factor")
  divisor : current.regs .fp32 [] "divfactor"=base.regs .fp32 [] "divfactor"
  a : current.regs .ptr [c.BM,c.BK] "A"=some (ap s c base t)
  b : current.regs .ptr [c.BK,c.BN] "B"=some (bp s c base t)

def HasAcc {α : Type} (c : Config) (value : TileIndex [c.BM,c.BN] → Int) (s : State α) : Prop :=
  s.regs .int [c.BM,c.BN] "acc_int"=some value

theorem preludeContext {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α) :
    let base:=(preludeExecution M q s c bias initial).val
    Context s c base 0 base ∧ HasAcc c (fun _=>0) base := by
  dsimp only
  constructor
  · constructor
    all_goals try rfl
    all_goals try simp only [ap_zero,bp_zero]
    all_goals simp [register,preludeExecution]
  · simp [HasAcc,preludeExecution]

def bodyExecution {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (base current : State α)
    (t : Nat) (old : TileIndex [c.BM,c.BN] → Int)
    (ctx : Context s c base t current) (acc : HasAcc c old current) :
    {final : State α // run M q s.M s.N (body s c bias)
      (current.setReg "k" .nat [] (fun _=>t))=some final} := by
  simp only [HasAcc] at acc
  by_cases even : s.K%(c.BK*c.S)=0
  all_goals
    refine ⟨?_,?_⟩
    swap
    simp [body,source,TypedSource.kernel,ComputeKernel.surfaceBody,run,step,arithmeticStep,addExpression,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,VeriTile.Triton.Region.cast,
      Option.bind,integralFloorDiv,integralMod,ctx.memory,ctx.rk,ctx.a,ctx.b,acc,even]
    rfl

theorem bodyContext {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (base current : State α)
    (t : Nat) (old : TileIndex [c.BM,c.BN] → Int)
    (ctx : Context s c base t current) (acc : HasAcc c old current) :
    let final:=(bodyExecution M q s c bias base current t old ctx acc).val
    Context s c base (t+1) final ∧ HasAcc c (update q s c base t old) final := by
  dsimp only
  by_cases even : s.K%(c.BK*c.S)=0
  all_goals
    constructor
    · constructor
      all_goals simp [bodyExecution,even,ctx.memory,ctx.pids,ctx.pm,ctx.pn,ctx.pz,ctx.rm,ctx.rn,
        ctx.rk,ctx.wx,ctx.xx,ctx.divisor,ap,bp,Nat.add_mul,Nat.one_mul,Nat.mul_one,Nat.add_assoc]
      all_goals funext i
      all_goals simp [bop,Broadcast.leftIndex,Broadcast.rightIndex,ap,bp,Nat.add_mul,Nat.add_assoc]
    · simp [HasAcc,bodyExecution,even,update,dotAdd,av,bv,ctx.pm,ctx.pn,ctx.pids,register,
        bop,Broadcast.leftIndex,Broadcast.rightIndex]
      funext i
      rcases i with ⟨i,j,u⟩
      cases u
      simp [dotAdd,av,bv,even,register,defaultValue]

#axiomsClean preludeContext
#axiomsClean bodyExecution
#axiomsClean bodyContext
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote

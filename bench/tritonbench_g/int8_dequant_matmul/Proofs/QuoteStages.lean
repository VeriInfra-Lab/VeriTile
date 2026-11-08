import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteSemantics
import VeriTile.Triton.Float.Integer

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote
open VeriTile.Triton FP.Structural
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false

def source (s : Shape) (c : Config) (bias : Bool) : ComputeKernel :=
  TypedSource.kernel "A" "B" "C" "bias" "state_x" "state_w"
    s.M s.N s.K s.K 1 s.N 1 s.N 1 c.BM c.BN c.BK 8 c.S (steps s c)
    (s.K%(c.BK*c.S)==0) bias
/-- cdiv on nonnegative caller dimensions agrees with the source's signed
ceil division; positivity follows from the existing launch configuration. -/
theorem signed_cdiv_nat (a b : Nat) (positive : 0<b) :
    ((a:Int)+((b:Int)-1))/(b:Int)=(((a+b-1)/b:Nat):Int) := by
  have h : (a:Int)+((b:Int)-1)=((a+b-1:Nat):Int) := by omega
  rw [h,Int.natCast_ediv]

theorem nat_lt_signed_sub (a b c : Nat) : ((a:Int)<(b:Int)-(c:Int)) ↔ a<b-c := by omega

def prelude (s : Shape) (c : Config) (bias : Bool) := (source s c bias).surfaceBody.take 32
def body (s : Shape) (c : Config) (bias : Bool) : List ComputeStmt :=
  match (source s c bias).surfaceBody[32]? with
  | some (ComputeStmt.forRangeInt _ _ _ _ code)=>code
  | _=>[]
def epilogue (s : Shape) (c : Config) (bias : Bool) := (source s c bias).surfaceBody.drop 33

def loopStatement (s : Shape) (c : Config) (bias : Bool) : ComputeStmt :=
  ((source s c bias).surfaceBody[32]?).getD (.effectMarker "missing source loop")

theorem source_decomposition (s : Shape) (c : Config) (bias : Bool) :
    (source s c bias).surfaceBody=
      prelude s c bias++[loopStatement s c bias]++epilogue s c bias := rfl

/-- The literal cdiv expression is retained in the program; its value is
connected to the mathematical step count separately. -/
def loopBound (s : Shape) (c : Config) : Op .int [] :=
  .tritonDiv .int .nil
    (.add .int .nil (.castNatToInt (.constNat s.K))
      (.sub .int .nil (.castNatToInt (.mul .nat .nil (.constNat c.BK) (.constNat c.S))) (.constInt 1)))
    (.castNatToInt (.mul .nat .nil (.constNat c.BK) (.constNat c.S)))

theorem loop_shape (s : Shape) (c : Config) (bias : Bool) :
    loopStatement s c bias = .forRangeInt "k" (.constInt 0) (loopBound s c) (.constNat 1) (body s c bias) := rfl

theorem loop_bound_value {α : Type} [Inhabited α] (M : Algebra α)
    (s : Shape) (c : Config) (state : State α) :
    evalOp M none (loopBound s c) state=some (fun _=>(steps s c:Int)) := by
  have positive : 0<c.BK*c.S := Nat.mul_pos c.positiveK c.positiveS
  have nonnegative' : ¬(s.K:Int)+((c.BK:Int)*c.S-1)<0 := by
    have h := Int.natCast_nonneg s.K
    have h' : (1:Int)≤(c.BK:Int)*c.S := by exact_mod_cast positive
    omega
  unfold loopBound
  rw [FP.Structural.eval_truncDiv]
  simp only [evalOp_unfold,numeric,bop,Broadcast.leftIndex,Broadcast.rightIndex,
    bind,pure,Option.bind]
  change some (fun _ => Int.tdiv ((s.K:Int)+((c.BK*c.S:Nat)-1)) (c.BK*c.S)) = _
  rw [Int.tdiv_eq_ediv_of_nonneg (by omega)]
  rw [←Int.natCast_mul]
  rw [signed_cdiv_nat s.K (c.BK*c.S) positive]
  rfl


theorem step_loop {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (state : State α) :
    step M q s.M s.N (loopStatement s c bias) state=
      rangeInt M q s.M s.N "k" 0 (steps s c) 1 (body s c bias) state := by
  rw [loop_shape,step]
  simp only [loop_bound_value]
  simp only [evalOp_unfold,bind,pure,Option.bind]

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α) :
    {final : State α // run M q s.M s.N (prelude s c bias) initial=some final} := by
  have nonnegativeM : ¬(s.M:Int)+((c.BM:Int)-1)<0 := by
    have := c.positiveM
    omega
  have nonnegativeN : ¬(s.N:Int)+((c.BN:Int)-1)<0 := by
    have := c.positiveN
    omega
  refine ⟨?_,?_⟩
  swap
  simp [prelude,source,TypedSource.kernel,TypedSource.sourceKernel,ComputeKernel.surfaceBody,run,step,arithmeticStep,addExpression,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,VeriTile.Triton.Region.cast,
    Option.bind,integralFloorDiv,integralMod,
    signed_cdiv_nat _ _ c.positiveM,signed_cdiv_nat _ _ c.positiveN,
    nonnegativeM,nonnegativeN,-Int.natCast_ediv]
  rfl

#axiomsClean loop_bound_value
#axiomsClean step_loop
#axiomsClean source_decomposition
#axiomsClean preludeExecution
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote

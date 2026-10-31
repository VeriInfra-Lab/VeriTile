import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteSemantics

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
def prelude (s : Shape) (c : Config) (bias : Bool) := (source s c bias).surfaceBody.take 30
def body (s : Shape) (c : Config) (bias : Bool) : List ComputeStmt :=
  match (source s c bias).surfaceBody[30]? with
  | some (ComputeStmt.forRange _ _ _ _ code)=>code
  | _=>[]
def epilogue (s : Shape) (c : Config) (bias : Bool) := (source s c bias).surfaceBody.drop 31

theorem source_decomposition (s : Shape) (c : Config) (bias : Bool) :
    (source s c bias).surfaceBody=
      prelude s c bias++[.forRange "k" 0 (steps s c) 1 (body s c bias)]++epilogue s c bias := rfl

def preludeExecution {α : Type} [Inhabited α] (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (bias : Bool) (initial : State α) :
    {final : State α // run M q s.M s.N (prelude s c bias) initial=some final} := by
  refine ⟨?_,?_⟩
  swap
  simp [prelude,source,TypedSource.kernel,ComputeKernel.surfaceBody,run,step,arithmeticStep,addExpression,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,VeriTile.Triton.Region.cast,
    Option.bind,integralFloorDiv,integralMod]
  rfl

#axiomsClean source_decomposition
#axiomsClean preludeExecution
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.Quote

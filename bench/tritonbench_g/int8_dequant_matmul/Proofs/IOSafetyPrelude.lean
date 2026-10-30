import bench.tritonbench_g.int8_dequant_matmul.Proofs.QuoteGeometry
import VeriTile.Triton.Float.IntegerDotSafety

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

theorem traceSafe_of_pointwise (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (bounds : RegionName → Nat) (code : List ComputeStmt)
    (hs : ∀ st∈code,∀ state : State α,IntegerDot.stepSafe M a context bounds st state)
    (state : State α) : IntegerDot.traceSafeList M a context bounds code state := by
  induction code generalizing state with
  | nil => exact IntegerDot.traceSafeList_nil M a context bounds state
  | cons st rest ih =>
    apply (IntegerDot.traceSafeList_cons M a context bounds st rest state).mpr
    refine ⟨hs st (by simp) state,?_⟩
    intro next _
    exact ih (fun stmt h => hs stmt (by simp [h])) next

def preFactors (s : Shape) (c : Config) (hasBias : Bool) : List ComputeStmt :=
  (prelude s c hasBias).take 27

/-- The original 27 statements before the first scale load manipulate only
registers and pointer expressions. Safety holds for arbitrary initial memory. -/
theorem preFactors_safe (M : Algebra α) (a : IntegerDot.Arithmetic)
    (context : IntegerDot.Context) (s : Shape) (c : Config) (hasBias : Bool)
    (state : State α) (bounds : RegionName → Nat) :
    IntegerDot.traceSafeList M a context bounds (preFactors s c hasBias) state := by
  apply traceSafe_of_pointwise
  simp [preFactors,prelude,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
    ComputeKernel.surfaceBody]
  all_goals simp only [IntegerDot.stepSafe_unfold,Safety.stepSafe_unfold,
    Safety.safeExpr_unfold,Safety.safeCompute_unfold]
  all_goals simp [Safety.safeOp_unfold]

def preFactorsExecution (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α) :
    {final : State α // Quote.run M q s.M s.N (preFactors s c hasBias) initial=some final} := by
  refine ⟨?_,?_⟩
  swap
  simp [preFactors,prelude,source,TypedSource.kernel,TypedSource.skeleton,TypedSource.bindRuntime,
    ComputeKernel.surfaceBody,Quote.run,Quote.step,arithmeticStep,addExpression,
    FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,store,
    numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
    FloatDType.numericDType,FloatDType.toTileDType,VeriTile.Triton.Region.cast,
    Option.bind,integralFloorDiv,integralMod]
  rfl

private theorem signed_rem (a b : Nat) :
    (a:Int)-(b:Int)*((a:Int)/(b:Int))=(a%b : Nat) := by
  rw [Int.natCast_emod]
  exact (Int.emod_def (a:Int) (b:Int)).symm

theorem prefix_ram (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<IntScaledMatmul.Revision001.Geometry.count (geometry s c)) :
    register (preFactorsExecution M q s c hasBias initial).val .int [c.BM] "ram"=
      (fun i : TileIndex [c.BM] => ((row s c (initial.pids 0) i.1%s.M : Nat):Int)) := by
  have pm := congrFun (prelude_pm M q s c hasBias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,preFactorsExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pm ⊢
  simp only [←neg_mul,←ite_mul] at pm ⊢
  rw [pm]
  have nr : ¬((blockM s c (initial.pids 0):Int)*c.BM+i.1.val)<0 := not_lt.mpr (by positivity)
  simp only [if_neg nr]
  exact signed_rem _ _

theorem prefix_rbn (M : Algebra α) (q : IntegerModel I F H)
    (s : Shape) (c : Config) (hasBias : Bool) (initial : State α)
    (valid : initial.pids 0<IntScaledMatmul.Revision001.Geometry.count (geometry s c)) :
    register (preFactorsExecution M q s c hasBias initial).val .int [c.BN] "rbn"=
      (fun i : TileIndex [c.BN] => ((col s c (initial.pids 0) i.1%s.N : Nat):Int)) := by
  have pn := congrFun (prelude_pn M q s c hasBias initial valid) PUnit.unit
  funext i
  simp [register,preludeExecution,preFactorsExecution,bop,Broadcast.leftIndex,Broadcast.rightIndex] at pn ⊢
  simp only [←neg_mul,←ite_mul] at pn ⊢
  rw [pn]
  have nr : ¬((blockN s c (initial.pids 0):Int)*c.BN+i.1.val)<0 := not_lt.mpr (by positivity)
  simp only [if_neg nr]
  exact signed_rem _ _

#axiomsClean preFactors_safe
#axiomsClean preFactorsExecution
#axiomsClean prefix_ram
#axiomsClean prefix_rbn
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.IOSafety

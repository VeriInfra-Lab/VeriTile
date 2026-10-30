import bench.tritonbench_g.int8_dequant_matmul.Proofs.PublicIO

namespace VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.ReuseIndices
open VeriTile Triton FP.Structural Quote
noncomputable section
set_option maxHeartbeats 12000000
set_option maxRecDepth 32000
set_option linter.unusedSimpArgs false
variable {α : Type} [Inhabited α]

def source (s : Shape) (c : Config) (bias : Bool) : ComputeKernel :=
  TypedSource.reuseIndicesKernel "A" "B" "C" "bias" "state_x" "state_w"
    s.M s.N s.K s.K 1 s.N 1 s.N 1 c.BM c.BN c.BK 8 c.S (steps s c)
    (s.K%(c.BK*c.S)==0) bias

def prefixCode (s : Shape) (c : Config) (bias : Bool) : List ComputeStmt :=
  (Quote.source s c bias).surfaceBody.take 25

def repeated (s : Shape) (c : Config) (bias : Bool) : List ComputeStmt :=
  ((Quote.source s c bias).surfaceBody.drop 25).take 2

def suffixCode (s : Shape) (c : Config) (bias : Bool) : List ComputeStmt :=
  (Quote.source s c bias).surfaceBody.drop 27

def headCode (s : Shape) (c : Config) (bias : Bool) : List ComputeStmt :=
  (Quote.source s c bias).surfaceBody.take 18

def segmentCode (s : Shape) (c : Config) (bias : Bool) : List ComputeStmt :=
  ((Quote.source s c bias).surfaceBody.drop 18).take 7

/-- Only the short index/pointer segment is unfolded. Earlier grouped-grid
arithmetic remains shared and opaque to this local register proof. -/
theorem segment_run (M : Algebra α) (q : IntegerModel I F H) (s : Shape) (c : Config)
    (bias : Bool) (initial : State α) :
    Quote.run M q s.M s.N (segmentCode s c bias ++ repeated s c bias) initial =
      Quote.run M q s.M s.N (segmentCode s c bias) initial := by
  cases hm : initial.regs .int [] "pid_m" <;>
    cases hn : initial.regs .int [] "pid_n" <;>
    cases hz : initial.regs .int [] "pid_z" <;>
    simp [segmentCode,repeated,Quote.source,TypedSource.kernel,
      ComputeKernel.surfaceBody,Quote.run,Quote.step,arithmeticStep,addExpression,
      FP.Structural.step,evalExpr,evalComputeOp,evalOp_unfold,evalOffsets,
      numeric,bop,natLt,numericLt,numericEq,ComputeDType.eraseDType,ofFloat,toFloat,
      FloatDType.numericDType,FloatDType.toTileDType,VeriTile.Triton.Region.cast,
      Option.bind,integralFloorDiv,integralMod,hm,hn,hz]
  simp only [State.setReg]
  congr 1
  funext d sh n
  by_cases hrm : n="rm" <;> by_cases hrn : n="rn" <;>
    simp [State.setReg,hrm,hrn]

theorem old_prefix_run (M : Algebra α) (q : IntegerModel I F H) (s : Shape) (c : Config)
    (bias : Bool) (initial : State α) :
    Quote.run M q s.M s.N (IOSafety.preFactors s c bias) initial =
      Quote.run M q s.M s.N (prefixCode s c bias) initial := by
  have old : IOSafety.preFactors s c bias =
      headCode s c bias ++ (segmentCode s c bias ++ repeated s c bias) := rfl
  have candidate : prefixCode s c bias = headCode s c bias ++ segmentCode s c bias := rfl
  rw [old,candidate,Quote.run_append,Quote.run_append]
  cases run M q s.M s.N (headCode s c bias) initial with
  | none => rfl
  | some state => exact segment_run M q s c bias state

theorem execute_eq (M : Algebra α) (q : IntegerModel I F H) (s : Shape) (c : Config)
    (bias : Bool) (initial : State α) :
    Quote.execute M q s.M s.N (source s c bias) initial =
      Quote.execute M q s.M s.N (Quote.source s c bias) initial := by
  have old : (Quote.source s c bias).surfaceBody=IOSafety.preFactors s c bias ++ suffixCode s c bias := rfl
  have candidate : (source s c bias).surfaceBody=prefixCode s c bias ++ suffixCode s c bias := rfl
  unfold Quote.execute
  rw [old,candidate,Quote.run_append,Quote.run_append,old_prefix_run]

theorem prefix_safe (M : Algebra α) (q : IntegerModel I F H) (s : Shape) (c : Config)
    (bias : Bool) (initial : State α) (bounds : RegionName → Nat) :
    IntegerDot.traceSafeList M (SharedInteger.arithmetic q) (SharedInteger.context q s.M s.N)
      bounds (prefixCode s c bias) initial := by
  have parts : IOSafety.preFactors s c bias = prefixCode s c bias ++ repeated s c bias := rfl
  have safe := IOSafety.preFactors_safe M (SharedInteger.arithmetic q)
    (SharedInteger.context q s.M s.N) s c bias initial bounds
  rw [parts,IntegerDot.traceSafeList_append] at safe
  exact safe.1

theorem safe_iff (M : Algebra α) (q : IntegerModel I F H) (s : Shape) (c : Config)
    (bias : Bool) (initial : State α) (bounds : RegionName → Nat) :
    IntegerDot.TraceSafe M (SharedInteger.arithmetic q) (SharedInteger.context q s.M s.N)
        bounds (source s c bias) initial ↔
      IntegerDot.TraceSafe M (SharedInteger.arithmetic q) (SharedInteger.context q s.M s.N)
        bounds (Quote.source s c bias) initial := by
  have old : (Quote.source s c bias).surfaceBody=IOSafety.preFactors s c bias ++ suffixCode s c bias := rfl
  have candidate : (source s c bias).surfaceBody=prefixCode s c bias ++ suffixCode s c bias := rfl
  have leftSafe := IOSafety.preFactors_safe M (SharedInteger.arithmetic q)
    (SharedInteger.context q s.M s.N) s c bias initial bounds
  have rightSafe := prefix_safe M q s c bias initial bounds
  unfold IntegerDot.TraceSafe
  rw [old,candidate,IntegerDot.traceSafeList_append,IntegerDot.traceSafeList_append]
  simp only [leftSafe,rightSafe,true_and]
  rw [SharedInteger.run_eq,SharedInteger.run_eq,old_prefix_run]

#axiomsClean segment_run
#axiomsClean execute_eq
#axiomsClean safe_iff
end
end VeriTile.Bench.Repairs.Int8DequantMatmul.ScalarCast.ReuseIndices

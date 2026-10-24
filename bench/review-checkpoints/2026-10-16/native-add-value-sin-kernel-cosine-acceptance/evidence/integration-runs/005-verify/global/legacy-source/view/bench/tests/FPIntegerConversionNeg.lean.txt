import VeriTile.Triton.DSL
import VeriTile.Triton.Float.IntegerConversion
import VeriTile.Triton.Float.AbsoluteValue
import VeriTile.Meta.StatementAudit

namespace VeriTile.Tests.FPIntegerConversionNeg
open VeriTile.Triton FP.Structural FP.IntegerConversion
noncomputable section
set_option maxHeartbeats 1600000
set_option linter.unusedSimpArgs false
local instance : Inhabited (FP.Value .fp32) := ⟨FP.Value.zero .fp32⟩

def allWidths (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32
  a = tl.load(x)
  i8 = a.to(tl.int8)
  i16 = a.to(tl.int16)
  i32 = a.to(tl.int32)
  i64 = a.to(tl.int64)
  u8 = a.to(tl.uint8)
  u16 = a.to(tl.uint16)
  u32 = a.to(tl.uint32)
  u64 = a.to(tl.uint64)
}
private def castDescriptor : ComputeStmt → Option (Bool × Nat × Option IndexComputePrecision × FloatDType)
  | .assign .int [] _ (.alg (.floatToInt p d w _)) => some (Bool.true, w, p, d)
  | .assign .nat [] _ (.alg (.floatToNat p d w _)) => some (Bool.false, w, p, d)
  | _ => none

/-- Every source integer spelling keeps its actual width, signedness and
floating compute precision. In particular, int32 does not become int8. -/
theorem allWidths_preserved (x : RegionName) :
    ((allWidths x).surfaceBody.drop 1).map castDescriptor =
      [some (Bool.true, 8, some .fp32, .real), some (Bool.true, 16, some .fp32, .real),
       some (Bool.true, 32, some .fp32, .real), some (Bool.true, 64, some .fp32, .real),
       some (Bool.false, 8, some .fp32, .real), some (Bool.false, 16, some .fp32, .real),
       some (Bool.false, 32, some .fp32, .real), some (Bool.false, 64, some .fp32, .real)] := rfl

def signedSource (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32, y = tl.int8
  a = tl.load(x)
  b = a.to(tl.int8)
  tl.store(y, b)
}
def unsignedSource (x y : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32, y = tl.uint8
  a = tl.load(x)
  b = tl.cast(a, tl.uint8)
  tl.store(y, b)
}

/-- Actual load/cast/store execution. The premise checks a decoded input
value, not the theorem's output or final execution. -/
theorem signed_source_execution (rest : Algebra (FP.Value .fp32))
    (x y : RegionName) (s : State (FP.Value .fp32)) (z : Int)
    (h : toSigned 8 ((s.mem x 0).read .real) = some z) :
    ∃ final, exec (fp32Backend (FP.Software.scalarFP32 rest)) (signedSource x y) s = some final ∧
      final.mem = (s.write y 0 (.mk .int z)).mem := by
  simp [signedSource, exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, fp32Backend, fp32Source, FP.Software.scalarFP32,
    State.setReg, store, TileShape.allIndices, ofFloat, toFloat,
    Option.bind, h, ComputeDType.eraseDType, State.write]

theorem unsigned_source_execution (rest : Algebra (FP.Value .fp32))
    (x y : RegionName) (s : State (FP.Value .fp32)) (z : Nat)
    (h : toUnsigned 8 ((s.mem x 0).read .real) = some z) :
    ∃ final, exec (fp32Backend (FP.Software.scalarFP32 rest)) (unsignedSource x y) s = some final ∧
      final.mem = (s.write y 0 (.mk .nat z)).mem := by
  simp [unsignedSource, exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, fp32Backend, fp32Source, FP.Software.scalarFP32,
    State.setReg, store, TileShape.allIndices, ofFloat, toFloat,
    Option.bind, h, ComputeDType.eraseDType, State.write]

def divisionSource (x y out : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32, y = tl.float32, out = tl.uint8
  a = tl.load(x)
  b = tl.load(y)
  c = (a / b).to(tl.uint8)
  tl.store(out, c)
}

/-- An integer result must not erase the source division's fp32 precision.
The frontend normalization removes redundant explicit precision scopes, so the
numeric conversion node itself must restore its recorded source context. -/
theorem division_source_execution (rest : Algebra (FP.Value .fp32)) (cfg : FP.Config)
    (x y out : RegionName) (s : State (FP.Value .fp32)) (z : Nat)
    (h : toUnsigned 8 (FP.div cfg ((s.mem x 0).read .real) ((s.mem y 0).read .real)) = some z) :
    ∃ final, exec (fp32Backend (FP.Software.scalarFP32 rest cfg))
        (divisionSource x y out) s = some final ∧
      final.mem = (s.write out 0 (.mk .nat z)).mem := by
  simp [divisionSource, exec, ComputeKernel.surfaceBody, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, fp32Backend, fp32Source, FP.Software.scalarFP32,
    FP.Software.fp32Context, numeric, bop, State.setReg, store,
    TileShape.allIndices, ofFloat, toFloat, Option.bind, h,
    ComputeDType.eraseDType, State.write]

/-- The untyped Real context is not silently interpreted as fp32. -/
theorem untagged_real_cast_fails (rest : Algebra (FP.Value .fp32))
    (s : State (FP.Value .fp32)) (a : Op .real []) :
    evalOp (fp32Backend rest) none (.floatToInt none .real 8 a) s = none := by
  rfl

/-- Missing support fails even when the tile has no lanes. -/
theorem missing_cast_fails_on_empty (rest : Algebra (FP.Value .fp32))
    (s : State (FP.Value .fp32)) (a : Op .real [0]) :
    evalOp (fp32Backend rest) none (.floatToNat (some .fp64) .real 8 a) s = none := by
  rfl

/-- A typed fp32 negative keeps the ordinary, precision-less context while
still reaching the real sign-bit implementation. -/
theorem typed_neg_execution (rest : Algebra (FP.Value .fp32))
    (s : State (FP.Value .fp32)) (x : FP.Value .fp32) :
    evalOp (FP.Software.scalarFP32 rest) none
      (.negFloat .fp32 (.ref .fp32 [] "x"))
      (s.setReg "x" .fp32 [] (fun _ => x)) = some (fun _ => x.neg) := by
  simp [evalOp_unfold, State.setReg, ofFloat, toFloat,
    FP.Software.scalarFP32, FP.Software.fp32Context]

-- Concrete sign-bit regressions distinguish fneg from subtraction by zero.
example : (FP.Value.zero .fp32).neg = FP.Value.zero .fp32 Bool.true := by decide +kernel
example : (FP.Value.zero .fp32 Bool.true).neg = FP.Value.zero .fp32 := by decide +kernel
example : (FP.Value.ofBits .fp32 0x7fa12345).neg = FP.Value.ofBits .fp32 0xffa12345 := by decide +kernel
example : (FP.Value.ofBits .fp32 0xffc54321).neg = FP.Value.ofBits .fp32 0x7fc54321 := by decide +kernel

/-- Abs clears the sign bit directly, including -0 and signaling NaNs. -/
theorem typed_abs_execution (rest : Algebra (FP.Value .fp32))
    (s : State (FP.Value .fp32)) (x : FP.Value .fp32) :
    evalOp (FP.Software.scalarFP32 rest) none
      (.absFloat .fp32 (.ref .fp32 [] "x"))
      (s.setReg "x" .fp32 [] (fun _ => x)) = some (fun _ => x.abs) := by
  simp [evalOp_unfold, State.setReg, ofFloat, toFloat,
    FP.Software.scalarFP32, FP.Software.fp32Context]

example : (FP.Value.zero .fp32 Bool.true).abs = FP.Value.zero .fp32 := by decide +kernel
example : (FP.Value.ofBits .fp32 0xffa12345).abs = FP.Value.ofBits .fp32 0x7fa12345 := by decide +kernel
example : (FP.Value.ofBits .fp32 0xffc54321).abs = FP.Value.ofBits .fp32 0x7fc54321 := by decide +kernel

def absSource (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32
  a = tl.load(x)
  b = tl.abs(a)
}
theorem abs_source_shape (x : RegionName) :
    (absSource x).surfaceBody[1]? = some
      (.assign .real [] "b" (.compute (.alg .fp32 (.absFloat .real (.ref .real [] "a"))))) := rfl

def genericAbsSource (dtype : FloatDType) (x : RegionName) : ComputeKernel :=
  triton {
  tl.region x = DTYPE
  a = tl.load(x)
  b = tl.abs(a)
}

theorem generic_abs_source_shape (dtype : FloatDType) (x : RegionName) :
    (genericAbsSource dtype x).surfaceBody[1]? = some
      (.assign dtype.toTileDType [] "b" (.alg (.absFloat dtype (.ref dtype.toTileDType [] "a")))) := by
  rfl

def negSource (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32
  a = tl.load(x)
  b = -a
}
theorem neg_source_shape (x : RegionName) :
    (negSource x).surfaceBody[1]? = some
      (.assign .real [] "b" (.compute (.alg .fp32 (.negFloat .real (.ref .real [] "a"))))) := rfl

def unarySources (x : RegionName) : ComputeKernel := triton {
  tl.region x = tl.float32
  a = tl.load(x)
  c = tl.cos(a)
  s = tl.sin(a)
  e = tl.exp2(a)
  l = tl.log2(a)
  t = tl.tanh(a)
}
private def fp32Assigned : ComputeStmt → Bool
  | .assign _ _ _ (.compute (.alg .fp32 _)) => Bool.true
  | _ => Bool.false

theorem unary_context_preserved (x : RegionName) :
    ((unarySources x).surfaceBody.drop 1).map fp32Assigned = [Bool.true, Bool.true, Bool.true, Bool.true, Bool.true] := rfl

#axiomsClean allWidths_preserved
#axiomsClean signed_source_execution
#axiomsClean unsigned_source_execution
#axiomsClean untagged_real_cast_fails
#axiomsClean missing_cast_fails_on_empty
#axiomsClean division_source_execution
#axiomsClean typed_abs_execution
#axiomsClean abs_source_shape
#axiomsClean generic_abs_source_shape
#axiomsClean typed_neg_execution
#axiomsClean neg_source_shape
#axiomsClean unary_context_preserved
end
end VeriTile.Tests.FPIntegerConversionNeg

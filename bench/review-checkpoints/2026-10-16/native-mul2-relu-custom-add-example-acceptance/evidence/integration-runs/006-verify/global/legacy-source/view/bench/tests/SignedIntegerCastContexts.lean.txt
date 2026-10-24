import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedIntegerCastContexts
open VeriTile.Triton

def nested (Seeds Offsets : RegionName) : ComputeKernel := triton {
  tl.region Seeds = tl.int64, Offsets = tl.int64
  a = tl.rand(tl.load(Seeds).to(tl.int32), tl.load(Offsets + tl.arange(0, 4)).to(tl.int32))
  b = tl.rand((tl.load(Seeds)).to(tl.int32), (tl.load(Offsets + tl.arange(0, 4))).to(tl.int32))
  c = tl.rand(tl.cast(tl.load(Seeds), tl.int32), tl.cast(tl.load(Offsets + tl.arange(0, 4)), tl.int32))
}

def statement (Seeds Offsets : RegionName) (name : RegName) : ComputeStmt :=
  .assign .real [4] name (.compute (.alg .fp32 (.rand ⟨.i32, .i32, 10⟩
    (Op.intToSigned 32 (.load .int (.region Seeds (.constNat 0)) .none .zero))
    (Op.intToSigned 32 (.load .int (.region Offsets (.arange 4)) .none .zero)))))

theorem all_forms (Seeds Offsets : RegionName) : (nested Seeds Offsets).surfaceBody =
    [statement Seeds Offsets "a", statement Seeds Offsets "b", statement Seeds Offsets "c"] := rfl

def conditional (Offsets : RegionName) : ComputeKernel := triton {
  tl.region Offsets = tl.int64
  if tl.load(Offsets).to(tl.int32) < $((0 : Int)) {
    x = $((1 : Int))
  }
}

theorem cond_signed (Offsets : RegionName) : (conditional Offsets).surfaceBody =
    [.ifThen (.lt .int .nil (Op.intToSigned 32
      (.load .int (.region Offsets (.constNat 0)) .none .zero)) (.constInt 0))
      [.assign .int [] "x" (.alg (.constInt 1))]] := rfl

def pointer (Offsets Out : RegionName) : ComputeKernel := triton {
  tl.region Offsets = tl.int64, Out = tl.int64
  raw = tl.load(Offsets)
  idx = raw.to(tl.int32)
  tl.store(Out + idx, idx)
}

theorem pointer_cast_before_nat (Offsets Out : RegionName) :
    (pointer Offsets Out).surfaceBody = [
      .assign .int [] "raw" (.alg (.load .int (.region Offsets (.constNat 0)) .none .zero)),
      .assign .int [] "idx" (.alg (Op.intToSigned 32 (.ref .int [] "raw"))),
      .store .int [] (.region Out (.castIntToNat (.ref .int [] "idx")))
        (.alg (.ref .int [] "idx")) .none] := rfl

#axiomsClean pointer_cast_before_nat
#axiomsClean all_forms
#axiomsClean cond_signed

def subtractLoaded (Left Right : RegionName) : ComputeKernel := triton {
  tl.region Left = tl.int32, Right = tl.int32
  right = tl.load(Right)
  difference = tl.load(Left) - right
}

/-- A loaded int32 register and an inline int32 load use the same signed
arithmetic channel. Storage metadata must not reject their subtraction. -/
theorem subtract_loaded_signed (Left Right : RegionName) :
    (subtractLoaded Left Right).surfaceBody = [
      .assign .int [] "right" (.alg (.load .int (.region Right (.constNat 0)) .none .zero)),
      .assign .int [] "difference" (.alg (.sub .int .nil
        (.load .int (.region Left (.constNat 0)) .none .zero)
        (.ref .int [] "right")))] := rfl

def subtractNaturalOffset (Left : RegionName) : ComputeKernel := triton {
  tl.region Left = tl.int32
  offset = tl.program_id(0)
  difference = tl.load(Left) - offset
}

/-- The natural program index is widened before subtraction, preserving
negative results until any later address conversion. -/
theorem subtract_loaded_natural_offset (Left : RegionName) :
    (subtractNaturalOffset Left).surfaceBody = [
      .assign .nat [] "offset" (.alg (.programId 0)),
      .assign .int [] "difference" (.alg (.sub .int .nil
        (.load .int (.region Left (.constNat 0)) .none .zero)
        (.castNatToInt (.ref .nat [] "offset"))))] := rfl

#axiomsClean subtract_loaded_signed
#axiomsClean subtract_loaded_natural_offset
end VeriTile.Tests.SignedIntegerCastContexts

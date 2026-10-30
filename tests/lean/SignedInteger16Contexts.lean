import VeriTile.Triton.DSL
import VeriTile.Meta.StatementAudit
namespace VeriTile.Tests.SignedInteger16Contexts
open VeriTile.Triton

def conditional (Offsets : RegionName) : ComputeKernel := triton {
  tl.region Offsets = tl.int64
  if tl.load(Offsets).to(tl.int16) < $((0 : Int)) {
    x = $((1 : Int))
  }
}

theorem cond_signed (Offsets : RegionName) : (conditional Offsets).surfaceBody =
    [.ifThen (.lt .int .nil (Op.intToSigned 16
      (.load .int (.region Offsets (.constNat 0)) .none .zero)) (.constInt 0))
      [.assign .int [] "x" (.alg (.constInt 1))]] := rfl

def pointer (Offsets Out : RegionName) : ComputeKernel := triton {
  tl.region Offsets = tl.int64, Out = tl.int64
  raw = tl.load(Offsets)
  idx = raw.to(tl.int16)
  tl.store(Out + idx, idx)
}

theorem pointer_cast_before_nat (Offsets Out : RegionName) :
    (pointer Offsets Out).surfaceBody = [
      .assign .int [] "raw" (.alg (.load .int (.region Offsets (.constNat 0)) .none .zero)),
      .assign .int [] "idx" (.alg (Op.intToSigned 16 (.ref .int [] "raw"))),
      .store .int [] (.region Out (.castIntToNat (.ref .int [] "idx")))
        (.alg (.ref .int [] "idx")) .none] := rfl

#axiomsClean pointer_cast_before_nat
#axiomsClean cond_signed

end VeriTile.Tests.SignedInteger16Contexts

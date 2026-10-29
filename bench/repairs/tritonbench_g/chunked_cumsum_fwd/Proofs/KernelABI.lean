import VeriTile.Triton.Float.BitValue
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.ABI
open VeriTile.Triton

/-- The ordinary Python `PyFloat_AsDouble` boundary. A successful result is
an unmodified binary64 word, including the sign and payload of a NaN. The
conversion-error branch covers Python type/overflow errors. No GPU result
is supplied by this host scalar boundary. -/
inductive Scalar where
  | converted (bits : BitVec 64)
  | conversionError (reason : String)
  deriving DecidableEq

/-- The pinned x86-64 default-rounding C double-to-float cast used by
Triton 3.7.1 driver.c extractFP32. Finite values round once to nearest-even,
without FTZ. NaNs retain sign/top payload and are quieted. This is an
explicit host ABI algorithm, not a deferred GPU numerical port. -/
def narrow64 (bits : BitVec 64) : BitVec 32 :=
  let negative := bits.toNat / 2^63 != 0
  let exponent : Nat := (bits.toNat / 2^52) % 2048
  let fraction : Nat := bits.toNat % 2^52
  if exponent = 2047 then
    if fraction = 0 then (FP.Value.infinity .fp32 negative).bits
    else BitVec.ofNat 32 ((if negative then 2^31 else 0) + 255*2^23 +
      ((fraction / 2^29) % 2^22) + 2^22)
  else
    let significand : Nat := if exponent = 0 then fraction else 2^52+fraction
    let scale : Int := if exponent = 0 then -1074 else (exponent : Int)-1075
    let magnitude := Rat.ofInt significand * FP.pow2 scale
    (FP.round .fp32 FP.Config.ieeeValue (if negative then -magnitude else magnitude) negative).bits

def extract : Scalar → Except String (BitVec 32)
  | .converted bits => .ok (narrow64 bits)
  | .conversionError reason => .error reason

def extractBounds (lower upper : Scalar) : Except String (BitVec 32 × BitVec 32) :=
  match extract lower with
  | .error reason => .error reason
  | .ok lo => match extract upper with
    | .error reason => .error reason
    | .ok hi => .ok (lo,hi)


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.ChunkedCumsumFwd.Kernel.ABI

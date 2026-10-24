import VeriTile.Triton.Float.ScalarOps
import VeriTile.Meta.StatementAudit

/-! Retained scalar representation lemma from the earlier proof. This optional
model computes binary64-to-FP32 conversion and the denominator's signed-zero
case. It does not assert native compiler conformance or create an experimental
obligation, and is not required by the integer-index equivalence. -/
namespace VeriTile.Bench.Repairs.DropoutTriton.ScalarABI
open VeriTile.Triton
abbrev Word := FP.Value .fp32

/-- Decode the actual Python floating-point word, retaining zero's sign. -/
def decodePython64 (word : BitVec 64) : FP.Decoded :=
  let negative := word.toNat / 2^63 != 0
  let exponent : Nat := word.toNat / 2^52 % 2048
  let fraction : Nat := word.toNat % 2^52
  if exponent == 2047 then
    if fraction == 0 then .infinity negative else .nan negative (fraction < 2^51) fraction
  else
    let significand : Nat := if exponent == 0 then fraction else 2^52 + fraction
    let scale : Int := (if exponent == 0 then -1022 else (exponent : Int)-1023)-52
    .finite negative (Rat.ofInt significand * FP.pow2 scale)

def pythonValue (word : BitVec 64) : Option Rat :=
  match decodePython64 word with
  | .finite sign magnitude => some (if sign then -magnitude else magnitude)
  | _ => none

def castPythonFP32 (word : BitVec 64) : Word :=
  match decodePython64 word with
  | .finite sign magnitude => FP.round .fp32 .ieeeValue (if sign then -magnitude else magnitude) sign
  | .infinity sign => FP.Value.infinity .fp32 sign
  | .nan sign _ payload => FP.Value.ofBits .fp32
      ((if sign then 2^31 else 0) + 255*2^23 + (payload/2^29 ||| 2^22))

def one : Word := FP.Value.ofBits .fp32 0x3f800000

def denominator (rawPython : BitVec 64) : Word :=
  FP.sub .ieeeValue one (castPythonFP32 rawPython)

theorem signed_zero_probability :
    castPythonFP32 0#64 = FP.Value.zero .fp32 Bool.false ∧
    castPythonFP32 0x8000000000000000#64 = FP.Value.zero .fp32 Bool.true ∧
    denominator 0#64 = denominator 0x8000000000000000#64 := by
  decide +kernel

#axiomsClean signed_zero_probability
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Repairs.DropoutTriton.ScalarABI

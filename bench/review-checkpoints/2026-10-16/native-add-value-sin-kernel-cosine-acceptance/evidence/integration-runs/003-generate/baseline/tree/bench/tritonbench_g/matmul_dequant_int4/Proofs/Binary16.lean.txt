import VeriTile.Triton.Float.BitValue
import VeriTile.Meta.StatementAudit
/-! Concrete IEEE binary16 software values. The shared scalar formats currently
contain only bf16/fp32; neither format is reused as binary16. This local profile
uses 1 sign, 5 exponent and 10 fraction bits, exponent bias 15, gradual underflow
at 2^-24, and direct exact-rational rounding. Arithmetic canonicalizes NaNs and
has no exception flags. It makes no GPU conformance claim. -/
namespace VeriTile.Bench.TritonBenchG.MatmulDequantInt4.Binary16
open VeriTile.Triton
structure Value where
  bits : BitVec 16
  deriving Repr, BEq, DecidableEq, Inhabited
namespace Value

def ofBits (n : Nat) : Value := ⟨BitVec.ofNat 16 n⟩
def raw (x : Value) : Nat := x.bits.toNat
def sign (x : Value) : Bool := x.raw / 32768 != 0
def exponent (x : Value) : Nat := (x.raw / 1024) % 32
def fraction (x : Value) : Nat := x.raw % 1024
private def pack (negative : Bool) (exp frac : Nat) : Value :=
  ofBits ((if negative then 32768 else 0) + exp * 1024 + frac)
def zero (negative : Bool := false) : Value := pack negative 0 0
def infinity (negative : Bool := false) : Value := pack negative 31 0
def maxFinite (negative : Bool := false) : Value := pack negative 30 1023
def qNaN : Value := pack false 31 512
def isSubnormal (x : Value) : Bool := x.exponent == 0 && x.fraction != 0

def decode (x : Value) : FP.Decoded :=
  let exp := x.exponent
  let frac := x.fraction
  if exp == 31 then
    if frac == 0 then .infinity x.sign else .nan x.sign (frac < 512) frac
  else
    let significand := if exp == 0 then frac else 1024 + frac
    let scale : Int := (if exp == 0 then -14 else (exp : Int) - 15) - 10
    .finite x.sign (Rat.ofInt significand * FP.pow2 scale)

/-- Pure sign-bit negation, including signed zero and NaN payloads. -/
def neg (x : Value) : Value :=
  ofBits ((if x.sign then 0 else 32768) + x.raw % 32768)

def toRat? (x : Value) : Option Rat :=
  match x.decode with
  | .finite negative magnitude => some (if negative then -magnitude else magnitude)
  | _ => none
end Value

private def roundMagnitude (mode : FP.Rounding) (negative : Bool) (q : Rat) : Nat :=
  let n := q.num.natAbs / q.den
  let rem := q.num.natAbs % q.den
  let up := match mode with
    | .nearestEven => 2 * rem > q.den || (2 * rem == q.den && n % 2 == 1)
    | .towardZero => false
    | .towardPositive => !negative && rem != 0
    | .towardNegative => negative && rem != 0
  if up then n + 1 else n

private def overflow (mode : FP.Rounding) (negative : Bool) : Value :=
  let toInfinity := match mode with
    | .nearestEven => true | .towardZero => false
    | .towardPositive => !negative | .towardNegative => negative
  if toInfinity then Value.infinity negative else Value.maxFinite negative

/-- Direct target-format rounding, with no intermediate fp32 value. -/
def round (cfg : FP.Config) (q : Rat) (negativeZero : Bool := false) : Value :=
  if q == 0 then Value.zero negativeZero
  else
    let negative := q < 0
    let magnitude := q.abs
    let estimate : Int := (magnitude.num.natAbs.log2 : Int) - magnitude.den.log2
    let exponent := if magnitude < FP.pow2 estimate then estimate - 1 else estimate
    let step : Int := max exponent (-14) - 10
    let significand := roundMagnitude cfg.rounding negative (magnitude / FP.pow2 step)
    let carry := significand >= 2048
    let significand := if carry then significand / 2 else significand
    let exponent := step + 10 + (if carry then 1 else 0)
    if exponent > 15 then overflow cfg.rounding negative
    else if significand < 1024 then
      if cfg.flushResults then Value.zero negative
      else Value.ofBits ((if negative then 32768 else 0) + significand)
    else Value.ofBits ((if negative then 32768 else 0) +
      (exponent + 15).toNat * 1024 + significand - 1024)

def prepare (cfg : FP.Config) (x : Value) : Value :=
  if cfg.flushInputs && x.isSubnormal then Value.zero x.sign else x

private def signed (negative : Bool) (magnitude : Rat) : Rat :=
  if negative then -magnitude else magnitude

def add (cfg : FP.Config) (x y : Value) : Value :=
  match (prepare cfg x).decode, (prepare cfg y).decode with
  | .nan .., _ | _, .nan .. => Value.qNaN
  | .infinity sx, .infinity sy => if sx == sy then Value.infinity sx else Value.qNaN
  | .infinity sx, _ => Value.infinity sx
  | _, .infinity sy => Value.infinity sy
  | .finite sx x, .finite sy y =>
    let negativeZero :=
      if x == 0 && y == 0 && sx == sy then sx else cfg.rounding == .towardNegative
    round cfg (signed sx x + signed sy y) negativeZero

def sub (cfg : FP.Config) (x y : Value) : Value := add cfg x y.neg

def mul (cfg : FP.Config) (x y : Value) : Value :=
  match (prepare cfg x).decode, (prepare cfg y).decode with
  | .nan .., _ | _, .nan .. => Value.qNaN
  | .infinity sx, .infinity sy => Value.infinity (sx != sy)
  | .infinity sx, .finite sy y =>
    if y == 0 then Value.qNaN else Value.infinity (sx != sy)
  | .finite sx x, .infinity sy =>
    if x == 0 then Value.qNaN else Value.infinity (sx != sy)
  | .finite sx x, .finite sy y =>
    round cfg (if sx != sy then -(x * y) else x * y) (sx != sy)

/-- Boundary vectors for gradual underflow, ties-to-even, overflow, signed
zero and NaN arithmetic, checked by Lean's kernel. -/
theorem ieee_boundary_vectors :
    round .ieeeValue (1 / 16777216) = Value.ofBits 0x0001 ∧
    round .ieeeValue (1 / 33554432) = Value.zero ∧
    round .ieeeValue (3 / 33554432) = Value.ofBits 0x0002 ∧
    round .ieeeValue 65504 = Value.ofBits 0x7bff ∧
    round .ieeeValue 65520 = Value.infinity ∧
    round .ieeeValue 0 true = Value.ofBits 0x8000 ∧
    mul .ieeeValue (Value.ofBits 0x8000) (Value.ofBits 0x3c00) = Value.ofBits 0x8000 ∧
    mul .ieeeValue Value.infinity Value.zero = Value.qNaN ∧
    mul .ieeeValue (Value.ofBits 0x7c01) (Value.ofBits 0x3c00) = Value.qNaN ∧
    (Value.ofBits 0x7c01).neg = Value.ofBits 0xfc01 ∧
    add .ieeeValue (Value.ofBits 0x3c00) (Value.ofBits 0x0c00) = Value.ofBits 0x3c00 ∧
    sub .ieeeValue (Value.ofBits 0x3c00) (Value.ofBits 0x0c00) = Value.ofBits 0x3c00 := by
  decide +kernel

attribute [spec_primitive] round add sub mul Value.neg
#axiomsClean ieee_boundary_vectors
end VeriTile.Bench.TritonBenchG.MatmulDequantInt4.Binary16

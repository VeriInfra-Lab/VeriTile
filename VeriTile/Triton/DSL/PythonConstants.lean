import VeriTile.Triton.DSL.Syntax
import Lean
import Init.Data.Rat.Basic

/-! Bounded Python literal arithmetic. Evaluate each floating operation with
binary64 nearest-even rounding, then check that one exact decimal literal has
the same binary64 value and FP32 conversion. The ideal Real projection is
preserved separately. No tensor operation is reassociated or evaluated here.
Zero/cancellation, nonfinite values and results outside normal FP32 are rejected
in this initial subset rather than silently becoming runtime FP32 arithmetic. -/
namespace VeriTile.Triton.DSL.PythonConstants
open Lean

private def pow2 (e : Int) : Rat :=
  if e < 0 then mkRat 1 (2 ^ e.natAbs) else Rat.ofInt (2 ^ e.toNat)

/-- Exact rational nearest-even rounding; `none` denotes overflow. -/
def roundBinary (precision : Nat) (minSub maxExp : Int) (q : Rat) : Option Rat := do
  if q == 0 then return 0
  let magnitude := q.abs
  let estimate : Int := (magnitude.num.natAbs.log2 : Int) - magnitude.den.log2
  let exponent := if magnitude < pow2 estimate then estimate - 1 else estimate
  let step := max minSub (exponent - (precision - 1 : Nat))
  let units := magnitude / pow2 step
  let floor := units.num.natAbs / units.den
  let rem := units.num.natAbs % units.den
  let up := 2 * rem > units.den || (2 * rem == units.den && floor % 2 == 1)
  let rounded := Rat.ofInt (if up then floor + 1 else floor) * pow2 step
  if rounded >= pow2 (maxExp + 1) then none
  else some (if q < 0 then -rounded else rounded)

private def round64 := roundBinary 53 (-1074) 1023
private def round32 := roundBinary 24 (-149) 127

structure Value where
  ideal : Rat
  binary64 : Rat
  floating : Bool
  deriving Repr

private def floatValue (q : Rat) : Except String Rat :=
  match round64 q with
  | some x => .ok x
  | none => .error "Python literal arithmetic overflows binary64"

private def combine (op : Rat → Rat → Rat) (a b : Value) : Except String Value := do
  let ideal := op a.ideal b.ideal
  if !a.floating && !b.floating then return ⟨ideal, ideal, Bool.false⟩
  let x ← if a.floating then pure a.binary64 else floatValue a.ideal
  let y ← if b.floating then pure b.binary64 else floatValue b.ideal
  let result ← floatValue (op x y)
  if x == 0 || y == 0 || result == 0 then
    throw "zero or cancellation in Python floating literal arithmetic is outside the checked subset"
  return ⟨ideal, result, Bool.true⟩

/-- `none` means a runtime/nonliteral expression, not an arithmetic error. -/
partial def evaluate (e : TSyntax `tritonExpr) : Except String (Option Value) := do
  match e with
  | `(tritonExpr| $n:num) => return some ⟨n.getNat, n.getNat, Bool.false⟩
  | `(tritonExpr| $n:scientific) =>
      let (m, denominator, exponent) := n.getScientific
      if exponent > 400 then throw "Python floating literal exponent is outside the checked bound"
      let q := if denominator then mkRat m (10 ^ exponent) else Rat.ofInt (m * 10 ^ exponent)
      return some ⟨q, ← floatValue q, Bool.true⟩
  | `(tritonExpr| ($inner:tritonExpr)) => evaluate inner
  | `(tritonExpr| -$inner:tritonExpr) =>
      return (← evaluate inner).map fun x => { x with ideal := -x.ideal, binary64 := -x.binary64 }
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) =>
      let some x ← evaluate a | return none
      let some y ← evaluate b | return none
      let operation := match e with
        | `(tritonExpr| $_:tritonExpr + $_:tritonExpr) => (· + ·)
        | `(tritonExpr| $_:tritonExpr - $_:tritonExpr) => (· - ·)
        | _ => (· * ·)
      return some (← combine operation x y)
  | _ => return none

private def arithmetic (e : TSyntax `tritonExpr) : Bool :=
  match e with
  | `(tritonExpr| $_:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr - $_:tritonExpr)
  | `(tritonExpr| $_:tritonExpr * $_:tritonExpr) => Bool.true
  | _ => Bool.false

/-- The resulting weak literal must encode the Python value, preserve its
normal-FP32 promotion, and retain the independent ideal Real expression. -/
def admissible (v : Value) : Bool :=
  v.floating && v.ideal != 0 &&
    round64 v.ideal == some v.binary64 && round32 v.ideal == round32 v.binary64 &&
    pow2 (-126) ≤ v.binary64.abs && v.binary64.abs ≤ (2 - pow2 (-23)) * pow2 127

private def decimal (q : Rat) : MacroM (TSyntax `tritonExpr) := do
  let mut scale := 1
  for exponent in [:401] do
    if scale % q.den == 0 then
      let mantissa := q.num.natAbs * (scale / q.den)
      let literal := Syntax.mkScientificLit (toString mantissa ++ "e-" ++ toString exponent)
      if q < 0 then return ← `(tritonExpr| -$literal:scientific)
      else return ← `(tritonExpr| $literal:scientific)
    scale := scale * 10
  Macro.throwError "Python constant has no checked terminating decimal representation"

partial def lower (raw : Syntax) : MacroM Syntax := do
  let e : TSyntax `tritonExpr := ⟨raw⟩
  if arithmetic e then
    match evaluate e with
    | .error message => Macro.throwErrorAt raw message
    | .ok (some value) =>
        if value.floating then
          unless admissible value do
            Macro.throwErrorAt raw "Python literal arithmetic cannot be folded preserving binary64, FP32 promotion and the ideal Real projection"
          return (← decimal value.ideal).raw
    | .ok none => pure ()
  match raw with
  | .node info kind args => return .node info kind (← args.mapM lower)
  | other => return other

end VeriTile.Triton.DSL.PythonConstants

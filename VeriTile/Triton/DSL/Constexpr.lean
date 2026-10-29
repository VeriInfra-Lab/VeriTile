import VeriTile.Triton.DSL.SourceDType

/-! A deliberately bounded static numeric subset. Exact binary64 powers of two
are admitted; arbitrary Python floating arithmetic and runtime powers are not
replaced by Real arithmetic. The source quotation retains its constexpr line. -/
namespace VeriTile.Triton.DSL.Constexpr
open Lean
syntax:80 tritonExpr:81 " ** " tritonExpr:80 : tritonExpr

private partial def integer (e : TSyntax `tritonExpr) : Option Int :=
  match e with
  | `(tritonExpr| $n:num) => some n.getNat
  | `(tritonExpr| ($x:tritonExpr)) => integer x
  | `(tritonExpr| -$x:tritonExpr) => (- ·) <$> integer x
  | `(tritonExpr| $a:tritonExpr + $b:tritonExpr) => return (← integer a) + (← integer b)
  | `(tritonExpr| $a:tritonExpr - $b:tritonExpr) => return (← integer a) - (← integer b)
  | `(tritonExpr| $a:tritonExpr * $b:tritonExpr) => return (← integer a) * (← integer b)
  | _ => none

/-- Parentheses delimit the original Python RHS without changing its value. -/
private partial def ungroup (e : TSyntax `tritonExpr) : TSyntax `tritonExpr :=
  match e with
  | `(tritonExpr| ($inner:tritonExpr)) => ungroup inner
  | _ => e

private def logPowerTwo (n : Nat) : Option Nat :=
  let k := n.log2
  if n == 2^k then some k else none

/-- Exact decimal spellings of positive binary powers only. Decimal fractions
that would need Python binary64 rounding fail closed in this subset. -/
private def floatPowerTwo (e : TSyntax `tritonExpr) : Option Int := do
  let `(tritonExpr| $n:scientific) := ungroup e | none
  let (m,denominator,exponent) := n.getScientific
  if exponent > 400 then none else do
    let num := if denominator then m else m * 10^exponent
    let den := if denominator then 10^exponent else 1
    let common := Nat.gcd num den
    let a ← logPowerTwo (num / common)
    let b ← logPowerTwo (den / common)
    let k := (a : Int) - b
    if -1074 ≤ k && k ≤ 1023 then some k else none

private def powerValue (e : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
  let `(tritonExpr| $base:tritonExpr ** $exponent:tritonExpr) := e
    | Macro.throwErrorAt e "expected a static floating power"
  let some k := floatPowerTwo base
    | Macro.throwErrorAt base "constexpr float power requires an exactly representable positive binary power literal"
  let some n := integer exponent
    | Macro.throwErrorAt exponent "constexpr power exponent must be a literal integer expression"
  unless -(2^53 : Int) ≤ n && n ≤ (2^53 : Int) do
    Macro.throwErrorAt exponent "constexpr integer exponent is outside exact binary64 conversion"
  let power := k*n
  unless -1074 ≤ power && power ≤ 1023 do
    Macro.throwErrorAt e "constexpr power is outside the finite nonzero binary64 range"
  let magnitude := Syntax.mkNumLit (toString power.natAbs)
  if power < 0 then `(tritonExpr| $((1 / ((2 : ℝ) ^ $magnitude:num) : ℝ)))
  else `(tritonExpr| $((((2 : ℝ) ^ $magnitude:num) : ℝ)))

/-- Only top-level exact static powers. Leave unsupported source constructs
intact so the normal parser/compiler rejects them. No runtime expression is
folded, and no statement is deleted or reordered. -/
def lower (body : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) :=
  body.mapM fun stmt => do
    match stmt with
    | `(tritonStmt| $name:ident : tl.constexpr = $value:tritonExpr) =>
        let value := ungroup value
        if let `(tritonExpr| $_base:tritonExpr ** $_exponent:tritonExpr) := value then
          let value ← powerValue value
          `(tritonStmt| $name:ident = $value:tritonExpr)
        else pure ⟨stmt⟩
    | _ => pure ⟨stmt⟩

end VeriTile.Triton.DSL.Constexpr

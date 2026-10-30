import VeriTile.Triton.DSL
import Lean.Meta.Transform

/-!
Explicit compatibility notation for historical mathematical int8 contracts.
`legacy_real_int8%` elaborates the displayed source, then replaces only its
signed-eight-bit numeric conversion with the old unbounded Real truncation
primitive. It also retains the historical Real comparison/subtraction spelling
of absolute value, which is mathematically equivalent but not a bitwise FP abs. This is a mathematical projection, not a typed FP execution or a
source-preserving cast rewrite. The real frontend and Structural evaluator are
unchanged. New FP specifications must never use this notation.

Keeping this boundary visible lets old mathematical statements remain checked
without silently giving the new typed float-to-integer operation a different
meaning. Widths other than eight and unsigned conversions are rejected.
-/
namespace VeriTile.Triton.LegacyRealInt8
open Lean Elab Term Meta

syntax (name := legacyRealInt8) "legacy_real_int8% " term : term

@[term_elab legacyRealInt8] def elabLegacyRealInt8 : TermElab := fun stx expected => do
  let original ← instantiateMVars (← elabTerm stx[1] expected)
  let zero ← elabTerm (← `(Op.const (0 : ℝ))) none
  let count ← IO.mkRef (0 : Nat)
  let projected ← Meta.transform original (post := fun e => do
    if e.isAppOfArity ``Op.absFloat 3 then
      let args := e.getAppArgs
      unless (← isDefEq args[1]! (mkConst ``FloatDType.real)) do
        throwError "legacy_real_int8% projects only the historical Real absolute-value expression"
      let value := args[2]!
      let shape ← whnf args[0]!
      unless shape.isAppOfArity ``List.cons 3 do
        throwError "legacy_real_int8% currently projects only nonempty-shape absolute value"
      let dims := shape.getAppArgs
      let scalarR := mkApp2 (mkConst ``Broadcast.scalarR) dims[1]! dims[2]!
      let scalarL := mkApp2 (mkConst ``Broadcast.scalarL) dims[1]! dims[2]!
      let cond ← mkAppM ``Op.lt #[mkConst ``ComparableDType.real, scalarR, value, zero]
      let neg ← mkAppM ``Op.sub #[mkConst ``NumericDType.real, scalarL, zero, value]
      return .done (← mkAppM ``Op.where #[cond, neg, value])
    if e.isAppOfArity ``Op.floatToNat 5 then
      throwError "legacy_real_int8% cannot project unsigned conversion"
    if e.isAppOfArity ``Op.floatToInt 5 then
      let args := e.getAppArgs
      unless (← getNatValue? args[3]!) == some 8 do
        throwError "legacy_real_int8% projects only an explicitly eight-bit signed conversion"
      let realValue ← mkAppM ``Op.castFloat #[args[2]!, mkConst ``FloatDType.real, args[4]!]
      let replacement ← mkAppM ``Op.castRealToInt8 #[realValue]
      count.modify (· + 1)
      return .done replacement
    return .done e)
  if (← count.get) == 0 then
    throwError "legacy_real_int8% expected an explicit signed-eight-bit conversion"
  return projected
end VeriTile.Triton.LegacyRealInt8

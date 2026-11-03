import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.IntegerWhereLiterals

/-! Optional support for Python float strings that denote exact binary64 integers.
The pasted source remains unchanged; only elaboration lowers the constructor to
its identical floating literal. Infinity still uses the existing bit encoding. -/
namespace VeriTile.Triton.DSL.ExactIntegerFloatStrings
open Lean Elab Term

/-- Exact integer-valued decimal strings within binary64's consecutive integer range.
Negative zero and fractional values are deliberately rejected. -/
def decimalInteger? (s : String) : Option String := do
  let negative := s.startsWith "-"
  let digits := if negative || s.startsWith "+" then (s.drop 1).toString else s
  let parts := digits.splitOn "."
  let whole ← match parts with
    | [n] => some n
    | [n, fraction] => if !fraction.isEmpty && fraction.toList.all (· == '0') then some n else none
    | _ => none
  if whole.isEmpty || !whole.toList.all (fun c => c >= '0' && c <= '9') then none else do
    let n ← whole.toNat?
    if n > 2^53 || (negative && n == 0) then none
    else some ((if negative then "-" else "") ++ toString n ++ ".0")

private partial def lower (raw : Syntax) : TermElabM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM lower)
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| float($value:term)) =>
    let `(term| $literal:str) := value | return raw
    if literal.getString == "inf" || literal.getString == "-inf" then return raw
    let some decimal := decimalInteger? literal.getString
      | throwErrorAt raw "finite float string must be an exact integer decimal with magnitude at most 2^53"
    match Parser.runParserCategory (← getEnv) `tritonExpr decimal with
    | .ok result => return result
    | .error message => throwErrorAt raw "{message}"
  | _ => return raw

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ k args => if k==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && !token.startsWith "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body
    | .error message => throwErrorAt raw "{message}"
  let parsed ← lower parsed
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete source body"
  let source := String.intercalate "\n" (stmts.toList.map (fun s=>render s.raw))
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  IntegerWhereLiterals.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.ExactIntegerFloatStrings

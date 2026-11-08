import VeriTile.Triton.DSL.PythonInlineSuites

/-! Integer-valued `tl.where` with two weak integer literals. Triton gives
these branches integer type independently of the condition's provenance. The
optional adapter makes that type explicit only inside elaboration; it leaves
mixed/floating branch expressions unchanged. The Int carrier is selected
without a narrowing operation: both source literals are checked to fit i32,
so the selected value is already in range (see IntegerWhereLiterals units). -/
namespace VeriTile.Triton.DSL.IntegerWhereLiterals
open Lean Elab Term

private partial def lower (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM lower)
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.where($condition:tritonExpr, $a:num, $b:num)) =>
    unless a.getNat<2^31 && b.getNat<2^31 do return raw
    return (← `(tritonExpr| tl.cast(tl.where($condition, tl.cast($a:num, tl.int64), tl.cast($b:num, tl.int64)), tl.int64))).raw
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
  let parsed ← liftMacroM (lower parsed)
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete source body"
  let source := String.intercalate "\n" (stmts.toList.map (fun s=>render s.raw))
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  CallSignature.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.IntegerWhereLiterals

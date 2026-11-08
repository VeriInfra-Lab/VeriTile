import VeriTile.Triton.DSL.PythonInlineSuites

/-! Parenthesize consecutive source method casts without removing either cast.
The stored Python quotation stays unchanged; this optional spelling adapter
feeds the ordinary dtype and precision elaborator. -/
namespace VeriTile.Triton.DSL.ChainedSourceCasts
open Lean Elab Term

scoped syntax:100 (name := castFloat) (priority := low) tritonExpr:100 ".to(" tritonDType ")" : tritonExpr
scoped syntax:100 (name := castQuery) (priority := low) tritonExpr:100 ".to(" ident ")" : tritonExpr

private def topCast (raw : Syntax) : Bool :=
  if raw.isOfKind ``castFloat || raw.isOfKind ``castQuery then Bool.true
  else if (methodCast? ⟨raw⟩).isSome then Bool.true
  else match (⟨raw⟩ : TSyntax `tritonExpr) with
    | `(tritonExpr| $method:ident($_value:tritonExpr)) =>
      match method.getId.eraseMacroScopes with
      | .str _ "to" => Bool.true
      | _ => Bool.false
    | _ => Bool.false

private partial def rewrite (raw : Syntax) : MacroM Syntax := do
  let receiver := raw[0]
  let chained := topCast receiver ||
    (receiver.isOfKind choiceKind && receiver.getArgs.any topCast)
  let raw := raw.setArgs (← raw.getArgs.mapM rewrite)
  unless chained do return raw
  if raw.isOfKind ``castFloat then
    let receiver : TSyntax `tritonExpr := ⟨raw[0]⟩
    let dtype : TSyntax `tritonDType := ⟨raw[2]⟩
    return (← `(tritonExpr| ($receiver:tritonExpr).to($dtype:tritonDType))).raw
  if raw.isOfKind ``castQuery then
    let receiver : TSyntax `tritonExpr := ⟨raw[0]⟩
    let dtype : Ident := ⟨raw[2]⟩
    return (← `(tritonExpr| ($receiver:tritonExpr).to($dtype:ident))).raw
  return raw

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args =>
    if kind==choiceKind then tokens args[0]!
    else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && !token.startsWith "." && token != "(" && token != "[" && previous != "." then
      out := out++" "
    out := out++token
    previous := token
  return out

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (Parser.parserExtension.activateScoped (← getEnv)
      `VeriTile.Triton.DSL.ChainedSourceCasts) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body
    | .error message => throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete source body"
  let stmts ← liftMacroM (stmts.mapM fun stmt => rewrite stmt.raw)
  let source := String.intercalate "\n" (stmts.toList.map render)
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
  CallSignature.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected

end VeriTile.Triton.DSL.ChainedSourceCasts

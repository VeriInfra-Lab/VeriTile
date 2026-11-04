import VeriTile.Triton.DSL.PastedRangeLoads

/-! Keep Python identifiers through repeated normalization of straight-line
bodies. The raw reader escapes `do` and `_`; its final invocation pass must
receive their original token spelling, rather than escaping them twice.
Only identifier tokens change parser spelling. Strings, operations, source
statements and the resulting register names remain unchanged. -/
namespace VeriTile.Triton.DSL.StraightLineIdentifiers
open Lean Elab Term

private partial def tokens : Syntax → List String
  | .atom _ value=>[value]
  | .ident _ spelling name _=>
    if ["do","_"].contains name.eraseMacroScopes.toString then [name.eraseMacroScopes.toString]
    else [spelling.toString]
  | .node _ kind args=>if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing=>[]
private def render (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw:=prepared.getArgs.back!
  let source:=raw.getAtomVal
  unless source.contains "«do»" || source.contains "«_»" do throwUnsupportedSyntax
  let suite:=(source.splitOn "\n").any fun line=>
    let text:=line.trimAscii.toString
    ["for ","if ","while ","else"].any (fun introducer=>text.startsWith introducer) && text.contains '{'
  if suite then throwUnsupportedSyntax
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++source) with
    | .ok body=>pure body | .error message=>throwErrorAt raw "{message}"
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete source body required"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (body.toList.map (render ·.raw))++"\n}")
  CallSignature.elaborateInvocation
    (prepared.setArgs (prepared.getArgs.set! (prepared.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.StraightLineIdentifiers

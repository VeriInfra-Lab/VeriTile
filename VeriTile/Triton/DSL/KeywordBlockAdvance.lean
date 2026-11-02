import VeriTile.Triton.DSL.PythonFullConstructors

/-! `tl.advance(pointer, offsets=(...))` and its positional spelling denote
exactly the same operation. Canonicalize only that keyword before pointer
storage inference; preserve the pointer expression, every offset and order. -/
namespace VeriTile.Triton.DSL.KeywordBlockAdvance
open Lean Elab Term
private partial def normalize (raw : Syntax) : MacroM Syntax := do
  let raw:=raw.setArgs (← raw.getArgs.mapM normalize)
  if let `(tritonExpr| tl.advance($pointer:tritonExpr, $key:ident=($offsets:tritonExpr,*))):=raw then
    unless key.getId.eraseMacroScopes==`offsets do
      Macro.throwErrorAt key "tl.advance only accepts the offsets keyword"
    return (← `(tritonExpr| tl.advance($pointer:tritonExpr, ($offsets:tritonExpr,*)))).raw
  return raw
private def prepare (stx : Syntax) : TermElabM Syntax := do
  let stx ← PythonFullConstructors.prepareSource stx
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let body ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body=>liftMacroM (normalize body) | .error message=>throwError "{message}"
  let `(triton { $stmts:tritonStmt* }):=body | throwError "complete function body required"
  let raw:=Syntax.atom .none ("{\n"++String.intercalate "\n"
    (stmts.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)
@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=> do
  CallSignature.elaborateInvocation (← prepare stx) expected
@[term_elab CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected=> do
  CallSignature.elaborateSignature (← prepare stx) expected
end VeriTile.Triton.DSL.KeywordBlockAdvance

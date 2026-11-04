import VeriTile.Triton.DSL.PythonFullConstructors
import VeriTile.Triton.DSL.EquivalentCallConstants
import VeriTile.Triton.DSL.FP32LoadedUpdates
import VeriTile.Triton.DSL.VerifiedStaticAssertions

/-! Python lists and tuples give the same ordered block-pointer offsets.
This opt-in reader changes only their delimiters before pointer dtype
tracking. It preserves pointer rebinding, nested loops and Python names. -/
namespace VeriTile.Triton.DSL.PythonBlockAdvances
open Lean Elab Term

private partial def lower (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower raw[0]!
  let raw:=raw.setArgs (← raw.getArgs.mapM lower)
  if let `(tritonExpr| tl.advance($pointer:tritonExpr, [$offsets:tritonExpr,*])):=raw then
    return (← `(tritonExpr| tl.advance($pointer:tritonExpr, ($offsets:tritonExpr,*)))).raw
  return raw

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

def prepare (stx : Syntax) : TermElabM Syntax := do
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok value=>liftMacroM (lower value)
    | .error message=>throwErrorAt raw "{message}"
  let parsed ← EquivalentCallConstants.prepare stx parsed
  let parsed ← FP32LoadedUpdates.prepare stx parsed
  let parsed ← VerifiedStaticAssertions.prepare stx parsed
  let `(triton { $body:tritonStmt* }):=parsed | throwError "complete source body required"
  let nested:=body.any (fun statement=>(tokens statement.raw).contains "{")
  let emit:=if nested then PythonFullConstructors.render else render
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (body.toList.map (emit ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

syntax (name := pythonBlockAdvanceCall) "with_python_block_advances " "in " term : term
@[term_elab pythonBlockAdvanceCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let call:=stx[2]
  unless call.isOfKind ``CallSignature.withCall do
    throwError "a complete source invocation is required"
  CallSignature.elaborateInvocation (← prepare call) expected
end VeriTile.Triton.DSL.PythonBlockAdvances

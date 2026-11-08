import VeriTile.Triton.DSL.ModuleFloatConstants

/-! Optional explicit-cast negative stop literals for descending Python ranges.
For representable nonnegative n, int64(-n) is exactly -int64(n). This only
exposes the existing signed reverse-range form, which captures bounds once
and mirrors the counter. No source statement or pointer update is removed. -/
namespace VeriTile.Triton.DSL.CastReverseRange
open Lean Elab Term Meta

private partial def lower (raw : Syntax) : MacroM Syntax := do
  if raw.isOfKind choiceKind then return ← lower raw[0]!
  let raw:=raw.setArgs (← raw.getArgs.mapM lower)
  if let `(tritonStmt| for $i:ident in range($start:tritonExpr, tl.cast(-$n:num, tl.int64), -$step:tritonExpr) { $body:tritonStmt* }) := raw then
    unless n.getNat<2^63 do Macro.throwErrorAt n "cast reverse-range stop must fit signed int64"
    return (← `(tritonStmt| for $i:ident in range($start:tritonExpr, -tl.cast($n:num, tl.int64), -$step:tritonExpr) { $body* })).raw
  return raw

private partial def tokens : Syntax → List String
  | .atom _ value=>[value]
  | .ident _ spelling _ _=>[spelling.toString]
  | .node _ k args=>if k==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing=>[]
private def render (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

def prepareSource (stx : Syntax) : TermElabM Syntax := do
  let stx ← ModuleFloatConstants.prepareSource stx
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>liftMacroM (lower parsed) | .error message=>throwError "{message}"
  let `(triton { $stmts:tritonStmt* }):=parsed | throwError "expected complete source body"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (stmts.toList.map (render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  PythonFullConstructors.elaborateInvocation (← prepareSource stx) expected
@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected=>do
  PythonFullConstructors.elaborateSignature (← prepareSource stx) expected
end VeriTile.Triton.DSL.CastReverseRange

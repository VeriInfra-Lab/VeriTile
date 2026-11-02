import VeriTile.Triton.DSL.PythonShapeTuples

/-! Python positional full constructors, constexpr RHS grouping and signed64
method casts use the existing typed forms. Normalization preserves shape,
values, dtype, casts and source statements. Unsupported typed constructors
continue to fail rather than being replaced by a value of another type. -/
namespace VeriTile.Triton.DSL.PythonFullConstructors
open Lean Elab Term Meta
syntax (name := positionalFull) "tl.full(" "[" tritonExpr,* "]" ", "
  tritonExpr ", " tritonDType ")" : tritonExpr

private partial def lower (raw : Syntax) : MacroM Syntax := do
  if raw.getKind == choiceKind then return ← lower raw[0]!
  -- A direct load followed by a pointer-element cast has a dedicated parser
  -- node. Group its receiver before pointer dtype resolution; otherwise the
  -- generic direct-load case can consume the load and discard this cast.
  let pointerLoadCast? ← match raw with
    | `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident)) =>
      pure (some (← `(tritonExpr| tl.load($p:tritonExpr, $mask:tritonExpr $[, $kwargs:tritonMemKwarg]*)),target))
    | `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($target:ident)) =>
      pure (some (← `(tritonExpr| tl.load($p:tritonExpr $[, $kwargs:tritonMemKwarg]*)),target))
    | _ => pure none
  if let some (operand,target) := pointerLoadCast? then
    match target.getId.eraseMacroScopes with
    | .str (.str _ "dtype") "element_ty" | .str (.str _ "type") "element_ty" =>
      let target : TSyntax `term := ⟨target.raw⟩
      return (← `(tritonExpr| ($operand:tritonExpr).to($target:term))).raw
    | _ => pure ()
  if raw.getKind == ``tritonDottedIdentMethodCast && raw.getNumArgs==4 then
    let callee:=raw[0].getId.eraseMacroScopes.toString
    let dt : TSyntax `tritonDType := ⟨raw[2]⟩
    let int64:=match dt with | `(tritonDType| tl.int64) => Bool.true | _ => Bool.false
    if callee.endsWith ".to" && int64 then
      let receiver:=mkIdent (callee.dropEnd 3).toString.toName
      let result ← `(tritonExpr| tl.cast($receiver:ident, $dt:tritonDType))
      return ← lower result.raw
  if raw.getKind == ``positionalFull then
    let `(tritonExpr| tl.full([$dims:tritonExpr,*], $value:tritonExpr, $dt:tritonDType)) := raw
      | Macro.throwError "expected a complete positional full constructor"
    let result ← `(tritonExpr| tl.full([$dims:tritonExpr,*], $value:tritonExpr, dtype=$dt:tritonDType))
    return ← lower result.raw
  return raw.setArgs (← raw.getArgs.mapM lower)

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args => if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
def render (raw : Syntax) : String := Id.run do
  let mut out:=""
  let mut previous:=""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

def prepareSource (stx : Syntax) : TermElabM Syntax := do
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let source:=String.intercalate "\n" ((source.splitOn "\n").map fun line=>
    match line.splitOn ": tl.constexpr =" with
    | [name,value] => if name.trimAscii.toString.toList.all (fun c=>c.isAlphanum || c=='_') then
        name++": tl.constexpr = ("++value++")" else line
    | _ => line)
  let source:=PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let body ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => liftMacroM (lower body) | .error message => throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }):=body | throwError "expected a complete function body"
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (stmts.toList.map (render ·.raw))++"\n}")
  return stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  CallSignature.elaborateInvocation (← prepareSource stx) expected
@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature]
def elaborateSignature : TermElab := fun stx expected => do
  CallSignature.elaborateSignature (← prepareSource stx) expected
end VeriTile.Triton.DSL.PythonFullConstructors

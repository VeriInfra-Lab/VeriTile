import VeriTile.Triton.DSL.FloatDTypeConstants
import VeriTile.Triton.DSL.PastedRangeLoads

/-! Resolve known floating dtype constexpr arguments in complete pasted bodies.
Compound and load receivers retain their complete expression and explicit
cast. Keyword names remain keys. Only actual dtype positions are specialized. -/
namespace VeriTile.Triton.DSL.ClosedFloatDTypes
open Lean Elab Term Meta

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwUnsupportedSyntax

private partial def lower (values : List (Name × TSyntax `tritonDType)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind == choiceKind then return ← lower values raw[0]!
  if raw.getKind == ``FloatDTypeConstants.namedFloatDType then
    let name := Name.mkSimple raw[0].getAtomVal
    let some (_,dtype) := values.find? (·.1 == name)
      | Macro.throwErrorAt raw "named dtype requires a known caller constexpr"
    return dtype.raw
  if raw.getKind == ``tritonMethodCastElementTy then
    let receiver : TSyntax `tritonExpr := ⟨raw[0]⟩
    let name := (methodCastTargetName? ⟨raw⟩).getD Name.anonymous
    if let some (_,dtype) := values.find? (·.1 == name) then
      let receiver : TSyntax `tritonExpr := ⟨← lower values receiver.raw⟩
      return (← `(tritonExpr| tl.cast($receiver:tritonExpr, $dtype:tritonDType))).raw
  if let `(tritonExpr| tl.load($pointer:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($id:ident)) := raw then
    if let some (_,dtype) := values.find? (·.1 == id.getId.eraseMacroScopes) then
      let result ← `(tritonExpr| tl.load($pointer:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dtype:tritonDType))
      return ← lower values result.raw
  if let `(tritonExpr| $callee:ident($arg:tritonExpr)) := raw then
    if callee.getId.eraseMacroScopes.toString.endsWith ".to" then
      if let `(tritonExpr| $id:ident) := arg then
        if let some (_,dtype) := values.find? (·.1 == id.getId.eraseMacroScopes) then
          let receiver := mkIdent (((callee.getId.eraseMacroScopes.toString.dropEnd 3).toString).toName)
          return (← `(tritonExpr| tl.cast($receiver:ident, $dtype:tritonDType))).raw
  match (⟨raw⟩ : TSyntax `tritonStmt) with
  | `(tritonStmt| $name:ident = $_:tritonExpr)
  | `(tritonStmt| $name:ident += $_:tritonExpr)
  | `(tritonStmt| $name:ident -= $_:tritonExpr)
  | `(tritonStmt| $name:ident *= $_:tritonExpr)
  | `(tritonStmt| $name:ident /= $_:tritonExpr)=>
    if values.any (·.1 == name.getId.eraseMacroScopes) then
      Macro.throwErrorAt name "dtype constexprs must remain immutable"
  | _ => pure ()
  let args := raw.getArgs
  let mut out := #[]
  for i in [:args.size] do
    -- The key of a keyword argument is not a use of the dtype value.
    if args[i]!.isIdent && args[i+1]?.any (fun next=>next.isAtom && next.getAtomVal.trimAscii.toString=="=") then
      out := out.push args[i]!
    else
      if args[i]!.isIdent && values.any (·.1 == args[i]!.getId.eraseMacroScopes) then
        Macro.throwErrorAt raw ("unsupported dtype position in " ++ toString raw.getKind ++ ": " ++ raw.reprint.getD "<no text>")
      out := out.push (← lower values args[i]!)
  if raw.isIdent && values.any (·.1 == raw.getId.eraseMacroScopes) then
    Macro.throwErrorAt raw "dtype constexpr used outside a cast dtype position"
  return raw.setArgs out

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected=>do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut types : List (Name × TSyntax `tritonDType) := []
  for item in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf item | throwUnsupportedSyntax
    let value ← elabTerm (mkIdent name.toName) none
    unless ← isDefEq (← inferType value) (mkConst ``FloatDType) do continue
    let value ← whnf value
    let kinds := [(``FloatDType.real,`tl.float64),(``FloatDType.fp32,`tl.float32),
      (``FloatDType.fp16,`tl.float16),(``FloatDType.bf16,`tl.bfloat16),
      (``FloatDType.f8e4,`tl.float8e4nv),(``FloatDType.f8e5,`tl.float8e5)]
    let some (_,dtype) := kinds.find? (fun pair=>value.isConstOf pair.1) | throwUnsupportedSyntax
    let some dtypeSyntax ← liftMacroM (dtypeSyntaxOfName? dtype) | throwUnsupportedSyntax
    types := types ++ [(name.toName,dtypeSyntax)]
  if types.isEmpty then throwUnsupportedSyntax
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw := prepared.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok value=>pure value | .error message=>throwErrorAt raw "{message}"
  let `(triton { $body:tritonStmt* }) := ← liftMacroM (lower types parsed)
    | throwError "complete source body required"
  let raw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
    (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}")
  let names ← mkListLit (mkConst ``String) (types.map (fun pair=>mkStrLit pair.1.toString))
  let profile ← exprToSyntax (← mkAppM ``FloatDTypeConstants.withoutTypes #[profile,names])
  let adjusted := prepared.setArgs ((prepared.getArgs.set! 1 profile).set! (prepared.getArgs.size-1) raw)
  elabTerm adjusted expected
end VeriTile.Triton.DSL.ClosedFloatDTypes

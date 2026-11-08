import VeriTile.Triton.DSL.FloatDTypeConstants

/-! Optional named FloatDType cast adapter, including load receivers.
Caller type metadata is specialized without changing runtime statements.
Each original load cast remains an explicit typed cast with its complete load.
No signed-stride normalization is performed by this adapter. -/
namespace VeriTile.Triton.DSL.NamedFloatDTypeCasts
open Lean Elab Term Meta

open FloatDTypeConstants (withoutTypes namedFloatDType fullPositionalType)

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "constexpr names must be statically known"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def lower (values : List (Name × TSyntax `tritonDType)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind == choiceKind then return ← lower values raw[0]!
  -- Expose cdiv's exact standard-library expression so signed-range
  -- inference also sees its operands; no integer narrowing is introduced.
  if let `(tritonExpr| tl.cdiv($a:tritonExpr, $b:tritonExpr)) := raw then
    let result ← `(tritonExpr| (($a:tritonExpr + ($b:tritonExpr - 1)) // $b:tritonExpr))
    return ← lower values result.raw
  -- The load-specific parser keeps the dtype argument as an identifier.
  -- Resolve only a declared FloatDType constexpr, retaining an explicit cast
  -- around the entire original load, including its mask and other value.
  if let `(tritonExpr| tl.load($pointer:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($id:ident)) := raw then
    if let some (_,dt) := values.find? (fun p=>p.1==id.getId.eraseMacroScopes) then
      let result ← `(tritonExpr| tl.load($pointer:tritonExpr $[, $kwargs:tritonMemKwarg]*).to($dt:tritonDType))
      return ← lower values result.raw
  if let `(tritonExpr| $callee:ident($argument:tritonExpr)) := raw then
    if callee.getId.eraseMacroScopes.toString.endsWith ".to" then
      if let `(tritonExpr| $id:ident) := argument then
        if let some (_,dt) := values.find? (fun p=>p.1==id.getId.eraseMacroScopes) then
          let value := mkIdent (((callee.getId.eraseMacroScopes.toString.dropEnd 3).toString).toName)
          let result ← `(tritonExpr| tl.cast($value:ident, $dt:tritonDType))
          return ← lower values result.raw
  if raw.getKind == ``fullPositionalType then
    let `(tritonExpr| tl.full([$dims:tritonExpr,*], $value:tritonExpr, $dt:tritonDType)) := raw
      | Macro.throwError "expected a complete positional full constructor"
    let value ← if dt.raw.getAtomVal == "tl.int64" || dt.raw[0]?.any (fun x=>x.getAtomVal=="tl.int64") then
      `(tritonExpr| tl.cast($value:tritonExpr, tl.int64)) else pure value
    let result ← `(tritonExpr| tl.full([$dims:tritonExpr,*], $value:tritonExpr, dtype=$dt:tritonDType))
    return ← lower values result.raw
  if raw.getKind == ``namedFloatDType then
    let id := raw[0]
    let some (_,bound) := values.find? (fun p=>p.1==Name.mkSimple id.getAtomVal)
      | Macro.throwErrorAt raw "a named dtype requires a declared FloatDType constexpr argument"
    return bound.raw
  let args := raw.getArgs.toList
  -- Assignment nodes, not keyword argument nodes, are recognized as statements.
  match (⟨raw⟩ : TSyntax `tritonStmt) with
  | `(tritonStmt| $id:ident = $_:tritonExpr)
  | `(tritonStmt| $id:ident := $_:tritonExpr)
  | `(tritonStmt| $id:ident += $_:tritonExpr) =>
    if values.any (fun p=>p.1==id.getId.eraseMacroScopes) then
      Macro.throwErrorAt id "FloatDType constexpr parameters must remain immutable"
  | _ => pure ()
  if let a::b::_ := args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
        (ids b).any (fun id=>values.any (fun p=>p.1==id)) then
      Macro.throwError "a loop cannot shadow a FloatDType constexpr argument"
  return raw.setArgs (← raw.getArgs.mapM (lower values))

/-- Reject an unhandled dtype value instead of letting a generic source-call
fallback erase its cast. Keyword names such as dtype= remain ordinary keys. -/
private partial def noDTypeValues (names : List Name) (raw : Syntax) : MacroM Unit := do
  if raw.isIdent && names.contains raw.getId.eraseMacroScopes then
    Macro.throwErrorAt raw "a FloatDType constexpr may occur only in a supported cast dtype position"
  let args:=raw.getArgs
  let leaf := match raw.getKind with | .str _ name => name | _ => ""
  let keywordOwner:=leaf.startsWith "tritonExpr" || leaf.startsWith "tritonMemKwarg" ||
    leaf.startsWith "tritonReduceKwarg"
  for i in [:args.size] do
    let key:=keywordOwner && args[i]!.isIdent && args[i+1]?.any
      (fun token=>token.isAtom && token.getAtomVal.trimAscii.toString=="=")
    unless key do noDTypeValues names args[i]!

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ k args => if k==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let mut types : List (Name × Expr) := []
  for raw in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf raw | throwError "constexpr name must be literal"
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``FloatDType) then types:=types++[(name.toName,value)]
  if types.isEmpty then return ← PythonShapeTuples.elaborateInvocation stx expected
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize
    (PythonInlineSuites.escapeNames (PythonShapeTuples.normalize source)))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body | .error message => throwErrorAt raw "{message}"
  let rec specialize (todo : List (Name × Expr))
      (values : List (Name × TSyntax `tritonDType)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let variants : List (TSyntax `tritonDType) := [← `(tritonDType| tl.float64),
        ← `(tritonDType| tl.float32), ← `(tritonDType| tl.float16),
        ← `(tritonDType| tl.bfloat16), ← `(tritonDType| tl.float8e4nv), ← `(tritonDType| tl.float8e5)]
      let branches ← variants.mapM fun dt=>specialize tail (values++[(name,dt)])
      let motive := mkLambda `dtype BinderInfo.default (mkConst ``FloatDType) (mkConst ``ComputeKernel)
      mkAppOptM ``FloatDType.casesOn (#[some motive,some value] ++ branches.toArray.map some)
    | [] =>
      let body ← liftMacroM (lower values parsed)
      let _ ← liftMacroM (noDTypeValues (types.map Prod.fst) body)
      let `(triton { $stmts:tritonStmt* }) := body | throwError "expected complete source body"
      let raw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (stmts.toList.map (render ·.raw))++"\n}")
      let names ← mkListLit (mkConst ``String) (types.map (fun p=>mkStrLit p.1.toString))
      let adjusted ← exprToSyntax (← mkAppM ``withoutTypes #[profile,names])
      let stx := stx.setArgs ((stx.getArgs.set! 1 adjusted.raw).set! (stx.getArgs.size-1) raw)
      CallSignature.elaborateInvocation stx expected
  specialize types []
end VeriTile.Triton.DSL.NamedFloatDTypeCasts

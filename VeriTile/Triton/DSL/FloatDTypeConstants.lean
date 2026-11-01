import VeriTile.Triton.DSL.PythonShapeTuples

/-! Immutable FloatDType constexpr arguments used in source cast positions.
They are caller type metadata, never runtime values or floating operations.
The optional adapter specializes every actual argument and only resolves its dtype
syntax; the original cast operation remains present. -/
namespace VeriTile.Triton.DSL.FloatDTypeConstants
open Lean Elab Term Meta

syntax (name := namedFloatDType) &"dtype" : tritonDType
syntax (name := fullPositionalType) "tl.full(" "[" tritonExpr,* "]" ", "
  tritonExpr ", " tritonDType ")" : tritonExpr

def withoutTypes (profile : CallSignature) (names : List String) : CallSignature :=
  {profile with constexpr := profile.constexpr.filter (!names.contains ·)}

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

private partial def usedSteps (raw : Syntax) : List Name :=
  let own:=match raw with
    | `(tritonStmt| for $_:ident in range($_:tritonExpr, $_:tritonExpr, $step:ident) { $_:tritonStmt* }) => [step.getId.eraseMacroScopes]
    | _ => []
  own ++ raw.getArgs.toList.flatMap usedSteps

private partial def immutable (name : Name) (raw : Syntax) : MacroM Unit := do
  let args:=raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i:=args.findIdx? (fun a=>a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
      if ((args.take i).flatMap ids).contains name then
        Macro.throwError "a constant signed stride must remain immutable"
  if let a::b::_:=args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString && (ids b).contains name then
      Macro.throwError "a loop cannot shadow a constant signed stride"
  for child in raw.getArgs do immutable name child

private partial def ranges (values : List (Name × TSyntax `num)) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:ident) { $body:tritonStmt* }) := raw then
    if let some (_,n):=values.find? (fun p=>p.1==stride.getId.eraseMacroScopes) then
      let body ← body.mapM fun st=>return (⟨← ranges values st⟩ : TSyntax `tritonStmt)
      return (← `(tritonStmt| for $i:ident in range(tl.cast($start:tritonExpr, tl.int64), tl.cast($stop:tritonExpr, tl.int64), tl.cast($n:num, tl.int64)) { $body* })).raw
  return raw.setArgs (← raw.getArgs.mapM (ranges values))

/-- Only a dominating, immutable, representable positive literal constructor
can supply a signed range step. Its original assignment remains present. -/
private def literalRanges (raw : Syntax) : MacroM Syntax := do
  let `(triton { $stmts:tritonStmt* }) := raw | return raw
  let steps:=usedSteps raw
  let mut active : List (Name × TSyntax `num):=[]
  let mut out:=#[]
  for st in stmts do
    let next ← ranges active st
    out:=out.push (⟨next⟩ : TSyntax `tritonStmt)
    if let `(tritonStmt| $id:ident = tl.full([], $n:num, tl.int64)) := st then
      unless steps.contains id.getId.eraseMacroScopes do continue
      unless 0<n.getNat && n.getNat<2^63 do Macro.throwError "signed stride literal must be positive and fit int64"
      for other in stmts do
        if other.raw != st.raw then immutable id.getId.eraseMacroScopes other
      active:=active++[(id.getId.eraseMacroScopes,n)]
  `(triton { $out:tritonStmt* })

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
      let body ← liftMacroM (literalRanges parsed)
      let body ← liftMacroM (lower values body)
      let _ ← liftMacroM (noDTypeValues (types.map Prod.fst) body)
      let `(triton { $stmts:tritonStmt* }) := body | throwError "expected complete source body"
      let raw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (stmts.toList.map (render ·.raw))++"\n}")
      let names ← mkListLit (mkConst ``String) (types.map (fun p=>mkStrLit p.1.toString))
      let adjusted ← exprToSyntax (← mkAppM ``withoutTypes #[profile,names])
      let stx := stx.setArgs ((stx.getArgs.set! 1 adjusted.raw).set! (stx.getArgs.size-1) raw)
      CallSignature.elaborateInvocation stx expected
  specialize types []
end VeriTile.Triton.DSL.FloatDTypeConstants

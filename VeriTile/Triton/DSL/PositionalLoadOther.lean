import VeriTile.Triton.DSL.PythonBitcastBooleans

/-! The three positional load arguments have exactly the same meaning as
pointer plus named mask/other. Closed Boolean metadata selects storage
branches before type checking; unresolved conditions remain in source control
flow. The pasted source remains unchanged. -/
namespace VeriTile.Triton.DSL.PositionalLoadOther
open Lean Elab Term Meta
syntax (name := positionalLoad) "tl.load(" tritonExpr ", " tritonExpr ", " tritonExpr
  ("," tritonMemKwarg)* ")" : tritonExpr
private partial def lower (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← lower raw[0]!
  if let `(tritonExpr| tl.load($ptr:tritonExpr, $mask:tritonExpr, $other:tritonExpr $[, $kwargs:tritonMemKwarg]*)) := raw then
    for kw in kwargs do
      let first := kw.raw[0]!
      let name := if first.isIdent then first.getId.toString else first.getAtomVal
      if ["mask","other"].contains name then
        Macro.throwErrorAt kw "load mask/other was supplied both positionally and by keyword"
    let result ← `(tritonExpr| tl.load($ptr:tritonExpr, mask=$mask:tritonExpr,
      other=$other:tritonExpr $[, $kwargs:tritonMemKwarg]*))
    return ← lower result.raw
  return raw.setArgs (← raw.getArgs.mapM lower)
private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "constexpr names must be a static list"
private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap identifiers
private partial def rejectWrites (names : List Name) (raw : Syntax) : MacroM Unit := do
  let args := raw.getArgs.toList
  for op in ["=",":=","+=","-=","*=","/=","%=","//="] do
    if let some i := args.findIdx? (fun t=>t.isAtom && t.getAtomVal.trimAscii.toString==op) then
      if (args.take i).any (fun a=>(identifiers a).any names.contains) then
        Macro.throwError "closed Boolean constexpr arguments must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for","tl.static_range"].contains first.getAtomVal.trimAscii.toString &&
        (identifiers counter).any names.contains then
      Macro.throwError "loop cannot shadow a closed Boolean constexpr"
  for child in raw.getArgs do rejectWrites names child
private partial def substitute (values : List (Name × Bool)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← substitute values raw[0]!
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_,value) := values.find? (fun p=>p.1==id.getId.eraseMacroScopes) then
      return (← if value then `(tritonExpr| true) else `(tritonExpr| false)).raw
  if raw.isIdent then
    if let some (_,value) := values.find? (fun p=>p.1==raw.getId.eraseMacroScopes) then
      return (← if value then `(tritonExpr| true) else `(tritonExpr| false)).raw
  return raw.setArgs (← raw.getArgs.mapM (substitute values))
private partial def selectClosed (raw : Syntax) : MacroM Syntax := do
  if let `(tritonStmt| if true { $body:tritonStmt* }) := raw then
    return ← selectClosed (Syntax.node .none `closedBooleanBlock (body.map (·.raw)))
  if let `(tritonStmt| if false { $body:tritonStmt* }) := raw then
    return Syntax.node .none `closedBooleanBlock #[]
  if let `(tritonStmt| if true { $yes:tritonStmt* } else { $no:tritonStmt* }) := raw then
    return ← selectClosed (Syntax.node .none `closedBooleanBlock (yes.map (·.raw)))
  if let `(tritonStmt| if false { $yes:tritonStmt* } else { $no:tritonStmt* }) := raw then
    return ← selectClosed (Syntax.node .none `closedBooleanBlock (no.map (·.raw)))
  let args ← raw.getArgs.mapM selectClosed
  if raw.isOfKind ``PythonBitcastBooleans.normalizedBitcast then
    let value : TSyntax `tritonExpr := ⟨args[0]!⟩
    let dtype : TSyntax `tritonDType := ⟨args[4]!⟩
    return (← `(tritonExpr| tl.bitcast($value, $dtype))).raw
  if raw.isOfKind ``PythonBitcastBooleans.normalizedDottedBitcast then
    let id := args[0]!.getId.eraseMacroScopes.toString
    unless id.endsWith ".to" do Macro.throwError "bitcast keyword requires a .to method"
    let value : TSyntax `tritonExpr := ⟨mkIdent ((id.dropEnd 3).toString.toName)⟩
    let dtype : TSyntax `tritonDType := ⟨args[2]!⟩
    return (← `(tritonExpr| tl.bitcast($value, $dtype))).raw
  return raw.setArgs (args.flatMap fun a=>if a.getKind==`closedBooleanBlock then a.getArgs else #[a])
@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut values : List (Name × Bool) := []
  let mut hasBooleans := Bool.false
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf entry | throwError "constexpr names must be literal strings"
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``Bool) then
      hasBooleans:=Bool.true
      let closed ← whnf value
      if closed.isConstOf ``Bool.true then values:=values++[(name.toName,Bool.true)]
      if closed.isConstOf ``Bool.false then values:=values++[(name.toName,Bool.false)]
  let stx ← PythonFullConstructors.prepareSource stx
  let raw := stx.getArgs.back!
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton "++raw.getAtomVal) with
    | .ok body => pure body | .error message => throwError "{message}"
  liftMacroM (rejectWrites (values.map Prod.fst) parsed)
  let parsed ← liftMacroM (lower parsed >>= substitute values >>= selectClosed)
  let `(triton { $body:tritonStmt* }) := parsed
    | throwError "complete body required"
  let source := "{\n"++String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))++"\n}"
  let stx := stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) (Syntax.atom raw.getHeadInfo source))
  if values.isEmpty then
    if hasBooleans then return ← PythonBitcastBooleans.elaborateInvocation stx expected
    else return ← PythonFullConstructors.elaborateInvocation stx expected
  let names ← mkListLit (mkConst ``String) (values.map fun p=>mkStrLit p.1.toString)
  let adjusted ← exprToSyntax (← mkAppM ``PythonBitcastBooleans.withoutBooleans #[profile,names])
  PythonFullConstructors.elaborateInvocation (stx.setArg 1 adjusted.raw) expected
end VeriTile.Triton.DSL.PositionalLoadOther

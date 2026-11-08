import VeriTile.Triton.DSL.OptionalPointers

/-! Immutable optional string constexpr parameters. Presence and equality
against literal strings are evaluated from the caller's parameter, outside
floating computation. Their original Python spelling stays in the quotation.
Other uses and writes are rejected instead of inventing a runtime string ABI. -/
namespace VeriTile.Triton.DSL.OptionalStringConstants
open Lean Elab Term Meta

def withoutOptions (profile : CallSignature) (names : List String) : CallSignature :=
  {profile with constexpr := profile.constexpr.filter (!names.contains ·)}

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "constexpr names must be statically known"

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids
private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for op in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun s=>s.isAtom && s.getAtomVal.trimAscii.toString==op) then
      unless ((args.take i).flatMap ids).all (!names.contains ·) do
        throwError "optional string constexpr parameters must remain immutable"
  if let a::b::_ := args then
    if a.isAtom && ["for","tl.for"].contains a.getAtomVal.trimAscii.toString &&
        (ids b).any names.contains then throwError "a loop cannot shadow an optional string constexpr"
  for arg in raw.getArgs do rejectWrites names arg

private partial def lower (values : List (Name × Ident)) (raw : Syntax) : MacroM Syntax := do
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| $id:ident is not None) =>
    if let some (_,value) := values.find? (fun p=>p.1==id.getId.eraseMacroScopes) then
      return (← `(tritonExpr| $((($value:ident).isSome : Bool)))).raw
  | `(tritonExpr| $id:ident is None) =>
    if let some (_,value) := values.find? (fun p=>p.1==id.getId.eraseMacroScopes) then
      return (← `(tritonExpr| $((!($value:ident).isSome : Bool)))).raw
  | `(tritonExpr| $id:ident == $text:str) =>
    if let some (_,value) := values.find? (fun p=>p.1==id.getId.eraseMacroScopes) then
      return (← `(tritonExpr| $((($value:ident == some $text:str) : Bool)))).raw
  | _ => pure ()
  if raw.isIdent && values.any (fun p=>p.1==raw.getId.eraseMacroScopes) then
    Macro.throwErrorAt raw "optional strings support only None guards and equality with a literal string"
  return raw.setArgs (← raw.getArgs.mapM (lower values))

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ k args => if k==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []
private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && !token.startsWith "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let mut options : List (Name × Expr) := []
  for name in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf name | throwError "constexpr name must be literal"
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkApp (mkConst ``Option [Level.zero]) (mkConst ``String)) then
      options:=options++[(name.toName,value)]
  if options.isEmpty then return ← PythonInlineSuites.elaborateInvocation stx expected
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body | .error message => throwErrorAt raw "{message}"
  rejectWrites (options.map Prod.fst) parsed
  let used := ids parsed ++ ids stx
  let rec freeze (todo : List (Name × Expr)) (values : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let mut fresh := Name.anonymous
      repeat
        let next ← mkFreshId
        fresh:=Name.mkSimple ("veritile_optional_string_"++(next.toString.map fun c=>if c.isAlphanum then c else '_'))
        unless used.contains fresh do break
      withLetDecl fresh (← inferType value) value fun bound => do
        let result ← freeze tail (values++[(name,mkIdent fresh)])
        mkLetFVars #[bound] result
    | [] =>
      let body ← liftMacroM (lower values parsed)
      let `(triton { $stmts:tritonStmt* }) := body | throwError "expected complete body"
      let source := String.intercalate "\n" (stmts.toList.map (fun s=>render s.raw))
      let raw := Syntax.atom raw.getHeadInfo ("{\n"++source++"\n}")
      let omitted ← mkListLit (mkConst ``String) (options.map (fun p=>mkStrLit p.1.toString))
      let adjusted ← exprToSyntax (← mkAppM ``withoutOptions #[profile,omitted])
      let stx := stx.setArgs ((stx.getArgs.set! 1 adjusted.raw).set! (stx.getArgs.size-1) raw)
      CallSignature.elaborateInvocation stx expected
  freeze options []
end VeriTile.Triton.DSL.OptionalStringConstants

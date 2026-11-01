import VeriTile.Triton.DSL.PythonInlineSuites

/-! Boolean constexpr control and immutable None mask aliases from copied
Python. Specialization selects only declared Boolean constexpr branches; all
runtime branches, loops and floating instructions remain. None aliases are
compile-time values, never fabricated Boolean registers. -/
namespace VeriTile.Triton.DSL.BooleanCallControl
open Lean Elab Term Meta

def withoutBooleans (profile : CallSignature) (names : List String) : CallSignature :=
  {profile with constexpr := profile.constexpr.filter (fun name => !names.contains name)}

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "constexpr names must be statically known"

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal text) ← whnf value | throwError "constexpr names must be static strings"
  return text

private partial def ids (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes] else raw.getArgs.toList.flatMap ids

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun x => x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      if (args.take i).any (fun x => (ids x).any names.contains) then
        throwError "Boolean constexpr arguments must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any names.contains then
      throwError "a loop cannot shadow a Boolean constexpr argument"
  for child in raw.getArgs do rejectWrites names child

private partial def substitute (values : List (Name × TSyntax `tritonExpr)) (raw : Syntax) : Syntax :=
  if let `(tritonExpr| $id:ident) := raw then
    match values.find? (fun p => p.1==id.getId.eraseMacroScopes) with
    | some (_,value) => value.raw
    | none => raw
  else if raw.isIdent then
    match values.find? (fun p => p.1==raw.getId.eraseMacroScopes) with
    | some (_,value) => value.raw
    | none => raw
  else raw.setArgs (raw.getArgs.map (substitute values))

private partial def boolean? (expression : TSyntax `tritonExpr) : Option Bool :=
  match expression with
  | `(tritonExpr| true) => some Bool.true
  | `(tritonExpr| false) => some Bool.false
  | `(tritonExpr| ($value:tritonExpr)) => boolean? value
  | `(tritonExpr| not $value:tritonExpr) => (!·) <$> boolean? value
  | `(tritonExpr| $left:tritonExpr and $right:tritonExpr) =>
      (· && ·) <$> boolean? left <*> boolean? right
  | `(tritonExpr| $left:tritonExpr or $right:tritonExpr) =>
      (· || ·) <$> boolean? left <*> boolean? right
  | _ => none

private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args => if kind==choiceKind then tokens args[0]! else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out:=out++" "
    out:=out++token
    previous:=token
  return out

private def noneExpr (value : TSyntax `tritonExpr) : Bool :=
  match value with
  | `(tritonExpr| None) => Bool.true
  | _ => Bool.false

/-- Runtime branches cannot establish compile-time None bindings. Loop-local
aliases stay local; a previous runtime assignment prevents a later None alias. -/
private partial def lower (values : List (Name × TSyntax `tritonExpr))
    (assigned : List Name) (runtime : Bool) (body : List (TSyntax `tritonStmt)) :
    MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | original::rest =>
    let args := original.raw.getArgs.toList
    for operator in ["=",":=","+=","-=","*=","/="] do
      if let some i := args.findIdx? (fun x => x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
        if (args.take i).any (fun x => (ids x).any (fun name => values.any (fun p => p.1==name))) then
          Macro.throwError "None mask aliases cannot be rebound"
    let stmt : TSyntax `tritonStmt := ⟨substitute values original.raw⟩
    match stmt with
    | `(tritonStmt| tl.store($pointer:tritonExpr, $value:tritonExpr, None)) =>
      let stmt ← `(tritonStmt| tl.store($pointer, $value))
      return stmt :: (← lower values (assigned++ids original.raw) runtime rest)
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      if let some selected := boolean? condition then
        return ← lower values assigned runtime ((if selected then yes else no).toList++rest)
      let yes ← lower values assigned Bool.true yes.toList
      let no ← lower values assigned Bool.true no.toList
      let stmt ← `(tritonStmt| if $condition { $yes.toArray* } else { $no.toArray* })
      return stmt :: (← lower values (assigned++ids original.raw) runtime rest)
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      if let some selected := boolean? condition then
        return ← lower values assigned runtime ((if selected then yes.toList else [])++rest)
      let yes ← lower values assigned Bool.true yes.toList
      let stmt ← `(tritonStmt| if $condition { $yes.toArray* })
      return stmt :: (← lower values (assigned++ids original.raw) runtime rest)
    | `(tritonStmt| for $index:ident in range($stop:tritonExpr) { $loop:tritonStmt* }) =>
      let loop ← lower values assigned runtime loop.toList
      let stmt ← `(tritonStmt| for $index in range(0, $stop) { $loop.toArray* })
      return stmt :: (← lower values (assigned++ids original.raw) runtime rest)
    | `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $loop:tritonStmt* }) =>
      let loop ← lower values assigned runtime loop.toList
      let stmt ← `(tritonStmt| for $index in range($start, $stop, $stride) { $loop.toArray* })
      return stmt :: (← lower values (assigned++ids original.raw) runtime rest)
    | `(tritonStmt| $name:ident = $rhs:tritonExpr) =>
      let name := name.getId.eraseMacroScopes
      if noneExpr rhs then
        if runtime || assigned.contains name || values.any (fun p => p.1==name) then
          Macro.throwError "None mask aliases must be fresh compile-time bindings"
        return ← lower (values++[(name,rhs)]) assigned runtime rest
      if values.any (fun p => p.1==name) then
        Macro.throwError "None mask aliases cannot be rebound"
      return stmt :: (← lower values (name::assigned) runtime rest)
    | _ =>
      return stmt :: (← lower values (assigned++ids original.raw) runtime rest)

@[term_elab CallSignature.withCall] def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← elabTermEnsuringType stx[1] (mkConst ``CallSignature)
  let mut booleans := []
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let name ← stringValue entry
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``Bool) then
      booleans:=booleans++[(name.toName,← exprToSyntax value)]
  if booleans.isEmpty then throwUnsupportedSyntax
  let source ← match RawSourceText.normalize stx.getArgs.back!.getAtomVal with
    | .ok source => pure source
    | .error message => throwError "{message}"
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed
    | .error message => throwError "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected complete source body"
  rejectWrites (booleans.map Prod.fst) parsed
  let names ← mkListLit (mkConst ``String) (booleans.map fun p => mkStrLit p.1.toString)
  let selected ← exprToSyntax (← mkAppM ``withoutBooleans #[profile,names])
  let rec emit (remaining : List (Name × TSyntax `term))
      (values : List (Name × TSyntax `tritonExpr)) : TermElabM (TSyntax `term) := do
    match remaining with
    | (name,value)::rest =>
      let yes ← emit rest (values++[(name,← `(tritonExpr| true))])
      let no ← emit rest (values++[(name,← `(tritonExpr| false))])
      `(if $value then $yes else $no)
    | [] =>
      let body ← liftMacroM <| lower values [] Bool.false stmts.toList
      let raw := Syntax.atom .none ("{\n"++String.intercalate "\n" (body.map (render ·.raw))++"\n}")
      let next := stx.setArg 1 selected.raw
      return ⟨next.setArgs (next.getArgs.set! (next.getArgs.size-1) raw)⟩
  elabTerm (← emit booleans []) expected

end VeriTile.Triton.DSL.BooleanCallControl

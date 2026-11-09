import VeriTile.Triton.DSL.PythonFullConstructors

/-! Immutable string constexpr parameters have no runtime register ABI. A
literal comparison partitions the caller's string into the literals actually
mentioned by the copied body and one remaining case. Each selected body keeps
all runtime statements, including branches and loops. In particular, pointer
aliases created by a constexpr branch are visible to subsequent type checking.
-/
namespace VeriTile.Triton.DSL.StringCallControl
open Lean Elab Term Meta

def withoutStrings (profile : CallSignature) (names : List String) : CallSignature :=
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
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some i := args.findIdx? (fun x => x.isAtom && x.getAtomVal.trimAscii.toString==operator) then
      if (args.take i).any (fun x => (ids x).any names.contains) then
        throwError "string constexpr parameters must remain immutable"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        (ids counter).any names.contains then
      throwError "a loop cannot shadow a string constexpr parameter"
  for child in raw.getArgs do rejectWrites names child

private def comparison? (names : List Name) (raw : Syntax) : Option (Name × String × Bool) := do
  let find (id : Ident) (text : TSyntax `str) (equal : Bool) := do
    let name := id.getId.eraseMacroScopes
    if names.contains name then some (name,text.getString,equal) else none
  match raw with
  | `(tritonExpr| $id:ident == $text:str)
  | `(tritonExpr| $text:str == $id:ident) => find id text Bool.true
  | `(tritonExpr| $id:ident != $text:str)
  | `(tritonExpr| $text:str != $id:ident) => find id text Bool.false
  | _ => none

private partial def comparisons (names : List Name) (raw : Syntax) : TermElabM (List (Name × String)) := do
  if raw.getKind==choiceKind then return ← comparisons names raw[0]!
  if let some (name,text,_) := comparison? names raw then return [(name,text)]
  if raw.isIdent && names.contains raw.getId.eraseMacroScopes then
    throwError "string constexpr parameters support only equality or inequality with a literal string"
  return (← raw.getArgs.toList.mapM (comparisons names)).flatten

private partial def boolean? (expression : TSyntax `tritonExpr) : Option Bool :=
  match expression with
  | `(tritonExpr| true) => some Bool.true
  | `(tritonExpr| false) => some Bool.false
  | `(tritonExpr| ($value:tritonExpr)) => boolean? value
  | `(tritonExpr| not $value:tritonExpr) => (!·) <$> boolean? value
  | `(tritonExpr| $a:tritonExpr and $b:tritonExpr) => (· && ·) <$> boolean? a <*> boolean? b
  | `(tritonExpr| $a:tritonExpr or $b:tritonExpr) => (· || ·) <$> boolean? a <*> boolean? b
  | _ => none

private partial def substitute (values : List (Name × Option String)) (raw : Syntax) : MacroM Syntax := do
  if raw.getKind==choiceKind then return ← substitute values raw[0]!
  if let some (name,text,equal) := comparison? (values.map Prod.fst) raw then
    let some (_,selected) := values.find? (·.1==name) | Macro.throwError "missing string specialization"
    let result := (selected==some text)==equal
    if result then return (← `(tritonExpr| true)).raw
    else return (← `(tritonExpr| false)).raw
  return raw.setArgs (← raw.getArgs.mapM (substitute values))

private partial def lower (body : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | stmt::rest =>
    match stmt with
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      if let some selected := boolean? condition then
        return ← lower ((if selected then yes else no).toList++rest)
      let yes ← lower yes.toList
      let no ← lower no.toList
      let stmt ← `(tritonStmt| if $condition { $yes.toArray* } else { $no.toArray* })
      return stmt::(← lower rest)
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      if let some selected := boolean? condition then
        return ← lower ((if selected then yes.toList else [])++rest)
      let yes ← lower yes.toList
      let stmt ← `(tritonStmt| if $condition { $yes.toArray* })
      return stmt::(← lower rest)
    | `(tritonStmt| for $index:ident in range(0, $stop:tritonExpr) { $loop:tritonStmt* }) =>
      let loop ← lower loop.toList
      let stmt ← `(tritonStmt| for $index in range(0, $stop) { $loop.toArray* })
      return stmt::(← lower rest)
    | `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr) { $loop:tritonStmt* }) =>
      let loop ← lower loop.toList
      let stmt ← `(tritonStmt| for $index in range($start, $stop) { $loop.toArray* })
      return stmt::(← lower rest)
    | `(tritonStmt| for $index:ident in range($stop:tritonExpr) { $loop:tritonStmt* }) =>
      let loop ← lower loop.toList
      let stmt ← `(tritonStmt| for $index in range($stop) { $loop.toArray* })
      return stmt::(← lower rest)
    | `(tritonStmt| for $index:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:tritonExpr) { $loop:tritonStmt* }) =>
      let loop ← lower loop.toList
      let stmt ← `(tritonStmt| for $index in range($start, $stop, $stride) { $loop.toArray* })
      return stmt::(← lower rest)
    | _ => return stmt::(← lower rest)

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``CallSignature))
  let mut strings : List (Name × Expr) := []
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf entry | throwError "constexpr name must be literal"
    let value ← elabTerm (mkIdent name.toName) none
    if ← isDefEq (← inferType value) (mkConst ``String) then
      strings:=strings++[(name.toName,value)]
  if strings.isEmpty then return ← PythonFullConstructors.elaborateInvocation stx expected
  -- Validate the original namespace before removing handled constexpr names.
  -- Otherwise a duplicate could disappear from the delegated ABI check.
  let mut names : List String := []
  for entry in ← listValues (← mkAppM ``CallSignature.constexpr #[profile]) do
    let .lit (.strVal name) ← whnf entry | throwError "constexpr name must be literal"
    names:=names++[name]
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers,
      ``CallSignature.scalars,``CallSignature.sourceConstants,``CallSignature.unaryImports] do
    for entry in ← listValues (← mkAppM field #[profile]) do
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[entry])
        | throwError "call binding names must be literal"
      names:=names++[name]
  unless names.eraseDups.length==names.length do throwError "duplicate name in call signature"
  let prepared ← PythonFullConstructors.prepareSource stx
  let raw := prepared.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source | .error message => throwError "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body | .error message => throwError "{message}"
  rejectWrites (strings.map Prod.fst) parsed
  let mentioned := (← comparisons (strings.map Prod.fst) parsed).eraseDups
  let partitions := strings.foldl (fun total entry =>
    total*((mentioned.filter (·.1==entry.1)).length+1)) 1
  unless partitions≤64 do throwError "too many string constexpr specializations"
  let omitted ← mkListLit (mkConst ``String) (strings.map (fun p=>mkStrLit p.1.toString))
  let adjusted ← exprToSyntax (← mkAppM ``withoutStrings #[profile,omitted])
  let rec specialize (todo : List (Name × Expr)) (values : List (Name × Option String)) : TermElabM Expr := do
    match todo with
    | [] =>
      let body ← liftMacroM (substitute values parsed)
      let `(triton { $statements:tritonStmt* }) := body | throwError "expected a complete source body"
      let statements ← liftMacroM (lower statements.toList)
      let selectedRaw := Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n"
        (statements.map (PythonFullConstructors.render ·.raw))++"\n}")
      let selected := prepared.setArgs ((prepared.getArgs.set! 1 adjusted.raw).set! (prepared.getArgs.size-1) selectedRaw)
      CallSignature.elaborateInvocation selected expected
    | (name,value)::tail =>
      let literals := (mentioned.filter (·.1==name)).map Prod.snd
      let rec alternatives (remaining : List String) : TermElabM Expr := do
        match remaining with
        | [] => specialize tail (values++[(name,none)])
        | literal::rest =>
          let selected ← exprToSyntax (← specialize tail (values++[(name,some literal)]))
          let fallback ← exprToSyntax (← alternatives rest)
          let value ← exprToSyntax value
          let literal := Syntax.mkStrLit literal
          elabTerm (← `(if $value == $literal then $selected else $fallback)) expected
      alternatives literals
  specialize strings []

end VeriTile.Triton.DSL.StringCallControl

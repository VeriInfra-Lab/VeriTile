import VeriTile.Triton.DSL.PythonInlineSuites

/-! Optional specialization of integer call arguments through explicit unsigned
self-casts. The original assignment remains in the program. Later reads use
its cast result, just as for signed self-casts in the base invocation frontend.
Caller expressions are frozen before source-local names enter scope. -/
namespace VeriTile.Triton.DSL.UnsignedCallArguments
open Lean Elab Term Meta

/-- An internal specialization profile; the user-facing signature is unchanged. -/
def withoutScalars (signature : Triton.CallSignature) (names : List String) : Triton.CallSignature :=
  {signature with scalars := signature.scalars.filter (fun entry => !names.contains entry.1)}

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "unsigned call profile must have statically known names"

private def stringValue (value : Expr) : TermElabM String := do
  let .lit (.strVal value) ← whnf value | throwError "call parameter name must be literal"
  return value

private def unsignedSelfCast? (raw : Syntax) : Option Name := do
  let `(tritonStmt| $target:ident = $rhs:tritonExpr) := raw | none
  let (receiver,dtype) ← methodCast? rhs <|> (match rhs with
    | `(tritonExpr| tl.cast($value:tritonExpr, $dtype:tritonDType)) => some (value,dtype)
    | _ => none)
  let name ← if receiver.raw.isIdent then some receiver.raw.getId.eraseMacroScopes
    else match receiver with
      | `(tritonExpr| $id:ident) => some id.getId.eraseMacroScopes
      | _ => none
  unless name == target.getId.eraseMacroScopes do none
  match dtype with
  | `(tritonDType| tl.uint8) | `(tritonDType| tl.uint16)
  | `(tritonDType| tl.uint32) | `(tritonDType| tl.uint64) => some name
  | _ => none

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers

private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some index := args.findIdx? (fun a => a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
      for id in args.take index do
        if id.isIdent && names.contains id.getId.eraseMacroScopes then
          throwErrorAt id "unsigned call specialization requires one top-level self-cast and no later mutation"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwErrorAt counter "a loop cannot shadow an unsigned call argument"
  for arg in raw.getArgs do rejectWrites names arg

private partial def substitute (values : List (Name × TSyntax `tritonExpr)) (raw : Syntax) : MacroM Syntax := do
  if raw.isIdent then
    if let some (_,value) := values.find? (fun p => p.1==raw.getId.eraseMacroScopes) then return value.raw
  if let `(tritonExpr| $id:ident) := raw then
    if let some (_,value) := values.find? (fun p => p.1==id.getId.eraseMacroScopes) then return value.raw
  if let some (receiver,dtype) := methodCast? ⟨raw⟩ then
    let next ← substitute values receiver.raw
    if next != receiver.raw then
      let receiver : TSyntax `tritonExpr := ⟨next⟩
      return (← `(tritonExpr| ($receiver).to($dtype:tritonDType))).raw
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (substitute values))
  | other => return other

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

@[term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed => pure parsed
    | .error message => throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete unsigned-argument body"
  let args := stx[4].getSepArgs
  let mut bindings : List (Name × Expr) := []
  for binding in args do
    let `(tritonCallBinding| $id:ident := $value:term) := binding | throwError "expected a named call argument"
    let name := id.getId.eraseMacroScopes
    if bindings.any (fun p => p.1==name) then throwErrorAt id "duplicate call argument binding"
    bindings:=bindings++[(name,← elabTerm value none)]
  let captured := (statements.toList.filterMap (unsignedSelfCast? ·.raw)).filter
    (fun name => bindings.any (fun p => p.1==name))
  if captured.isEmpty then return ← PythonInlineSuites.elaborateInvocation stx expected
  unless captured.eraseDups.length==captured.length do throwError "duplicate unsigned argument self-cast"
  let profile ← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature)
  let mut names : List String := []
  for field in [``Triton.CallSignature.pointers,``Triton.CallSignature.floatingPointers,
      ``Triton.CallSignature.scalars,``Triton.CallSignature.sourceConstants,``Triton.CallSignature.unaryImports] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      names:=names++[← stringValue (← mkAppM ``Prod.fst #[pair])]
  names:=names++(← (← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile])).mapM stringValue)
  unless names.eraseDups.length==names.length do throwError "duplicate name in call signature"
  let scalars ← listValues (← mkAppM ``Triton.CallSignature.scalars #[profile])
  for name in captured do
    let mut integral := Bool.false
    for pair in scalars do
      if (← stringValue (← mkAppM ``Prod.fst #[pair]))==name.toString then
        let kind ← whnf (← mkAppM ``Prod.snd #[pair])
        integral:=kind.isConstOf ``ScalarType.integer || kind.isConstOf ``ScalarType.i32 || kind.isConstOf ``ScalarType.i64
    unless integral do throwError "unsigned self-cast must bind an integer runtime parameter"
  for statement in statements do
    if (unsignedSelfCast? statement.raw).any captured.contains then continue
    rejectWrites captured statement.raw
  let used := identifiers parsed ++ identifiers stx
  let rec bind (todo : List (Name × Expr)) (frozen : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let dtype ← inferType value
      let mut localName := Name.anonymous
      repeat
        let fresh ← mkFreshId
        localName:=Name.mkSimple ("veritile_unsigned_argument_"++(fresh.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains localName do break
      withLetDecl localName dtype value fun bound => do
        let result ← bind tail (frozen++[(name,mkIdent localName)])
        mkLetFVars #[bound] result
    | [] =>
      let mut values : List (Name × TSyntax `tritonExpr) := []
      let mut kept : Array (TSyntax `tritonCallBinding) := #[]
      for (name,id) in frozen do
        if captured.contains name then
          let some (_,value) := bindings.find? (fun p => p.1==name)
            | throwError "missing unsigned call argument"
          let dtype ← inferType value
          let expression ← if ← isDefEq dtype (mkConst ``Nat) then
            liftMacroM `(tritonExpr| $((VeriTile.Triton.DSL.CallValue.natural $id:ident : Nat)))
          else if ← isDefEq dtype (mkConst ``Int) then
            liftMacroM `(tritonExpr| $((VeriTile.Triton.DSL.CallValue.integer $id:ident : Int)))
          else throwError "unsigned self-cast argument must have a Nat or Int carrier"
          values:=values++[(name,expression)]
        else
          let target := mkIdent name
          kept:=kept.push (← `(tritonCallBinding| $target:ident := $id:ident))
      let mut body : List (TSyntax `tritonStmt) := []
      for statement in statements do
        if let some name := unsignedSelfCast? statement.raw then
          if captured.contains name then
            let `(tritonStmt| $target:ident = $rhs:tritonExpr) := statement | throwError "expected self-cast"
            let rhs : TSyntax `tritonExpr := ⟨← liftMacroM <| substitute values rhs.raw⟩
            body:=body++[← `(tritonStmt| $target:ident = $rhs:tritonExpr)]
            values:=values.map (fun entry => if entry.1==name then (name,rhs) else entry)
            continue
        body:=body++[⟨← liftMacroM <| substitute values statement.raw⟩]
      let strings : Array (TSyntax `term) := captured.toArray.map (fun name => ⟨Syntax.mkStrLit name.toString⟩)
      let sig : TSyntax `term := ⟨stx[1]⟩
      let sig ← `(term| withoutScalars $sig [$strings,*])
      let call ← `(term| with_triton_call $sig call_args [$kept,*] in triton {})
      let raw := Syntax.atom .none ("{\n"++String.intercalate "\n" (body.map (render ·.raw))++"\n}")
      let call := call.raw.setArgs (call.raw.getArgs.set! (call.raw.getArgs.size-1) raw)
      PythonInlineSuites.elaborateInvocation call expected
  bind bindings []

end VeriTile.Triton.DSL.UnsignedCallArguments

import VeriTile.Triton.DSL.PythonInlineSuites

/-! Lower immutable signed scalar strides constructed from natural constexpr
values. The full assignment and all register reads remain; only a range's
step operand is constant-propagated after its dominating tl.full definition.
Runtime, negative, rebound and shadowed strides are not approximated. -/
namespace VeriTile.Triton.DSL.StaticSignedStrides
open Lean Elab Term Meta

syntax (name := withStaticCall) "with_triton_call " term " call_args "
  "[" tritonCallBinding,* "]" " static_strides " "in " "triton " RawSourceText.body : term

private partial def listValues (value : Expr) : TermElabM (List Expr) := do
  let value ← whnf value
  if value.isAppOfArity ``List.nil 1 then return []
  if value.isAppOfArity ``List.cons 3 then
    return value.getAppArgs[1]! :: (← listValues value.getAppArgs[2]!)
  throwError "constexpr profile must have statically known names"

private partial def identifiers (raw : Syntax) : List Name :=
  if raw.isIdent then [raw.getId.eraseMacroScopes]
  else raw.getArgs.toList.flatMap identifiers

/-- Propagation is valid only while both the source register and its
constexpr constructor remain immutable, including under nested scopes. -/
private partial def rejectWrites (names : List Name) (raw : Syntax) : TermElabM Unit := do
  let args := raw.getArgs.toList
  for operator in ["=",":=","+=","-=","*=","/="] do
    if let some index := args.findIdx? (fun a => a.isAtom && a.getAtomVal.trimAscii.toString==operator) then
      for id in args.take index do
        if id.isIdent && names.contains id.getId.eraseMacroScopes then
          throwErrorAt id "static_strides requires immutable stride and constexpr bindings"
  if let first::counter::_ := args then
    if first.isAtom && ["for","tl.for"].contains first.getAtomVal.trimAscii.toString &&
        counter.isIdent && names.contains counter.getId.eraseMacroScopes then
      throwErrorAt counter "a loop cannot shadow an immutable stride binding"
  for arg in raw.getArgs do rejectWrites names arg

/-- Separate tokens explicitly; the default formatter for a repeated custom
statement category need not insert spaces or newlines between statements. -/
private partial def tokens : Syntax → List String
  | .atom _ value => [value]
  | .ident _ spelling _ _ => [spelling.toString]
  | .node _ kind args =>
    if kind==choiceKind then tokens args[0]!
    else args.toList.flatMap tokens
  | .missing => []

private def render (raw : Syntax) : String := Id.run do
  let mut out := ""
  let mut previous := ""
  for token in tokens raw do
    if !out.isEmpty && token != "." && token != "(" && token != "[" && previous != "." then out := out++" "
    out := out++token
    previous := token
  return out


private def definition? (st : Syntax) : Option (Ident × Ident) := do
  let `(tritonStmt| $id:ident = tl.full((), $value:ident, tl.int64)) := st | none
  return (id,value)

private partial def replaceRanges (values : List (Name × Ident)) (declared : List Name) (raw : Syntax) : MacroM Syntax := do
  if let `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, $stride:ident) { $body:tritonStmt* }) := raw then
    if let some (_,value) := values.find? (fun p => p.1 == stride.getId.eraseMacroScopes) then
      let body : Array (TSyntax `tritonStmt) ← body.mapM fun s => return ⟨← replaceRanges values declared s⟩
      return (← `(tritonStmt| for $i:ident in range(tl.cast($start:tritonExpr, tl.int64), tl.cast($stop:tritonExpr, tl.int64), $value:ident) { $body* })).raw
    else if declared.contains stride.getId.eraseMacroScopes then
      Macro.throwErrorAt stride "static signed stride definition must dominate its range"
  match raw with
  | .node info kind args => return .node info kind (← args.mapM (replaceRanges values declared))
  | other => return other

@[term_elab withStaticCall] def elaborate : TermElab := fun stx expected => do
  let profile ← instantiateMVars (← elabTermEnsuringType stx[1] (mkConst ``Triton.CallSignature))
  let mut constants : List Name := []
  for entry in ← listValues (← mkAppM ``Triton.CallSignature.constexpr #[profile]) do
    let entry ← whnf entry
    let .lit (.strVal name) := entry | throwError "constexpr parameter name must be literal"
    constants := constants ++ [Name.mkSimple name]
  let raw := stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source => pure source
    | .error message => throwErrorAt raw "{message}"
  let source := PythonCallSyntax.normalize
    (PythonInlineSuites.normalize (PythonInlineSuites.escapeNames source))
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok body => pure body
    | .error message => throwErrorAt raw "{message}"
  let `(triton { $stmts:tritonStmt* }) := parsed | throwError "expected a complete source body"
  let mut bindings : List (Name × Expr) := []
  for stmt in stmts do
    if let some (name,value) := definition? stmt then
      unless constants.contains value.getId.eraseMacroScopes do continue
      if constants.contains name.getId.eraseMacroScopes then
        throwErrorAt name "a signed stride cannot rebind its constexpr parameter"
      if bindings.any (fun p => p.1 == name.getId.eraseMacroScopes) then
        throwErrorAt name "duplicate immutable signed stride definition"
      let bound ← elabTermEnsuringType value (mkConst ``Nat)
      -- Source constants and the stride register must stay immutable. Its
      -- own dominating definition is the only write removed from this check.
      for other in stmts do
        if other.raw == stmt.raw then continue
        rejectWrites [name.getId.eraseMacroScopes,value.getId.eraseMacroScopes] other
      bindings := bindings ++ [(name.getId.eraseMacroScopes,bound)]
  unless !bindings.isEmpty do throwError "static_strides requires a signed tl.full stride from a natural constexpr"
  let used := identifiers parsed ++ identifiers stx
  let rec bind (todo : List (Name × Expr)) (values : List (Name × Ident)) : TermElabM Expr := do
    match todo with
    | (name,value)::tail =>
      let mut fresh := Name.anonymous
      repeat
        let id ← mkFreshId
        fresh := Name.mkSimple ("veritile_static_stride_" ++
          (id.toString.map fun c => if c.isAlphanum then c else '_'))
        unless used.contains fresh do break
      withLetDecl fresh (mkConst ``Nat) value fun bound => do
        let result ← bind tail (values ++ [(name,mkIdent fresh)])
        mkLetFVars #[bound] result
    | [] =>
      let mut active : List (Name × Ident) := []
      let mut lowered : Array Syntax := #[]
      for stmt in stmts do
        let mut next ← liftMacroM <| replaceRanges active (values.map Prod.fst) stmt
        if let some (name,source) := definition? stmt then
          if let some value := values.find? (fun p => p.1 == name.getId.eraseMacroScopes) then
            -- Keep the declared signed scalar dtype even when its constexpr
            -- constructor value has a natural caller representation.
            next ← liftMacroM <| return (← `(tritonStmt|
              $name:ident = tl.full((), tl.cast($source:ident, tl.int64), tl.int64))).raw
            active := active ++ [value]
        lowered := lowered.push next
      let text := "{\n" ++ String.intercalate "\n" (lowered.toList.map render) ++ "\n}"
      let body := Syntax.atom .none text
      let base := mkNode ``CallSignature.withCall #[stx[0],stx[1],stx[2],stx[3],stx[4],stx[5],stx[7],stx[8],body]
      PythonInlineSuites.elaborateInvocation base expected
  bind bindings []

end VeriTile.Triton.DSL.StaticSignedStrides

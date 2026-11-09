import VeriTile.Triton.DSL.PythonFullConstructors

/-! Implicit floating atomic operand conversion from caller pointer metadata.
The pasted statement is unchanged. Pointer aliases retain their original
parameter's element dtype; unsupported destinations and control-flow merges
are rejected before the ordinary complete source frontend runs. -/
namespace VeriTile.Triton.DSL.ImplicitAtomicOperands
open Lean Elab Term Meta
private abbrev Aliases := List (Name × Name)

private partial def listValues (e : Expr) : TermElabM (List Expr) := do
  let e ← whnf e
  if e.isAppOfArity ``List.nil 1 then return []
  if e.isAppOfArity ``List.cons 3 then
    return e.getAppArgs[1]! :: (← listValues e.getAppArgs[2]!)
  throwError "atomic pointer metadata must be a static list"

private def pointerNames (profile : Syntax) : TermElabM (List Name × List Name) := do
  let profile ← elabTermEnsuringType profile (mkConst ``CallSignature)
  let mut all : List Name := []
  let mut floating : List Name := []
  for field in [``CallSignature.pointers,``CallSignature.floatingPointers] do
    for pair in ← listValues (← mkAppM field #[profile]) do
      let .lit (.strVal name) ← whnf (← mkAppM ``Prod.fst #[pair])
        | throwError "atomic pointer name must be literal"
      all := name.toName :: all
      let dtype ← whnf (← mkAppM ``Prod.snd #[pair])
      if field==``CallSignature.floatingPointers ||
          [``Triton.SourceDType.fp16,``Triton.SourceDType.bf16,``Triton.SourceDType.fp32,
           ``Triton.SourceDType.fp64,``Triton.SourceDType.f8e4,``Triton.SourceDType.f8e5].any dtype.isAppOf then
        floating := name.toName :: floating
  return (all,floating)

private partial def base (aliases : Aliases) (e : TSyntax `tritonExpr) : Option Name :=
  match e with
  | `(tritonExpr| $id:ident) => aliases.lookup id.getId.eraseMacroScopes
  | `(tritonExpr| ($x:tritonExpr)) => base aliases x
  | `(tritonExpr| $x:tritonExpr + $_:tritonExpr)
  | `(tritonExpr| $x:tritonExpr - $_:tritonExpr) => base aliases x
  | _ => none

private partial def containsAtomic (raw : Syntax) : Bool :=
  (raw.isAtom && raw.getAtomVal.startsWith "tl.atomic_add(") ||
    (raw.isIdent && raw.getId.eraseMacroScopes==`tl.atomic_add) || raw.getArgs.any containsAtomic

private partial def rejectWrites (names : List Name) (raw : Syntax) : MacroM Unit := do
  let args:=raw.getArgs
  if args.size≥2 && args[0]!.isIdent && args[1]!.isAtom &&
      ["=",":=","+=","-=","*=","/="].contains args[1]!.getAtomVal.trimAscii.toString &&
      names.contains args[0]!.getId.eraseMacroScopes then
    Macro.throwErrorAt raw "atomic pointer aliases cannot be rebound under control flow"
  if args.size≥2 && args[0]!.isAtom && args[0]!.getAtomVal.trimAscii.toString=="for" &&
      args[1]!.isIdent && names.contains args[1]!.getId.eraseMacroScopes then
    Macro.throwErrorAt raw "atomic pointer aliases cannot be rebound under control flow"
  if let `(tritonStmt| $a:ident, $b:ident $[, $more:ident]* = $_:tritonExpr, $_:tritonExpr $[, $_:tritonExpr]*) := raw then
    if (#[a,b]++more).any (fun id=>names.contains id.getId.eraseMacroScopes) then
      Macro.throwErrorAt raw "atomic pointer aliases cannot be rebound under control flow"
  for child in args do rejectWrites names child

private partial def lower (formals : List Name) (aliases : Aliases)
    (body : List (TSyntax `tritonStmt)) : MacroM (List (TSyntax `tritonStmt)) := do
  match body with
  | [] => return []
  | statement::tail =>
    let mut statement:=statement
    let mut next:=aliases
    match statement with
    | `(tritonStmt| $id:ident = $value:tritonExpr) =>
      if containsAtomic value.raw then Macro.throwErrorAt value "atomic return values are unsupported"
      let name:=id.getId.eraseMacroScopes
      if formals.contains name then Macro.throwErrorAt id "atomic pointer parameters must remain immutable"
      next:=aliases.filter (·.1!=name)
      if let some pointer:=base aliases value then next:=(name,pointer)::next
    | `(tritonStmt| tl.atomic_add($ptr:tritonExpr, $value:tritonExpr $[, $kwargs:tritonMemKwarg]*)) =>
      if containsAtomic ptr.raw || containsAtomic value.raw || kwargs.any (fun arg=>containsAtomic arg.raw) then
        Macro.throwErrorAt statement "nested atomic expressions are unsupported"
      let some pointer:=base aliases ptr
        | Macro.throwErrorAt ptr "cannot determine floating atomic destination dtype"
      let dtype : TSyntax `term:=⟨mkIdent (pointer.str "dtype" |>.str "element_ty")⟩
      statement ← `(tritonStmt| tl.atomic_add($ptr:tritonExpr, ($value:tritonExpr).to($dtype:term) $[, $kwargs:tritonMemKwarg]*))
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* } else { $no:tritonStmt* }) =>
      if containsAtomic condition.raw then Macro.throwErrorAt condition "atomic conditions are unsupported"
      for st in yes++no do rejectWrites (formals++aliases.map Prod.fst) st.raw
      let yes:=(← lower formals aliases yes.toList).toArray
      let no:=(← lower formals aliases no.toList).toArray
      statement ← `(tritonStmt| if $condition:tritonExpr { $yes* } else { $no* })
    | `(tritonStmt| if $condition:tritonExpr { $yes:tritonStmt* }) =>
      if containsAtomic condition.raw then Macro.throwErrorAt condition "atomic conditions are unsupported"
      for st in yes do rejectWrites (formals++aliases.map Prod.fst) st.raw
      let yes:=(← lower formals aliases yes.toList).toArray
      statement ← `(tritonStmt| if $condition:tritonExpr { $yes* })
    | `(tritonStmt| for $id:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $inner:tritonStmt* }) =>
      if containsAtomic a.raw || containsAtomic b.raw || containsAtomic c.raw then
        Macro.throwErrorAt statement "atomic loop bounds are unsupported"
      rejectWrites (formals++aliases.map Prod.fst) statement.raw
      let inner:=(← lower formals aliases inner.toList).toArray
      statement ← `(tritonStmt| for $id:ident in range($a:tritonExpr, $b:tritonExpr, $c:tritonExpr) { $inner* })
    | _ =>
      rejectWrites (formals++aliases.map Prod.fst) statement.raw
      if containsAtomic statement.raw then Macro.throwErrorAt statement "unsupported atomic control flow"
    return statement::(← lower formals next tail)

@[term_elab CallSignature.withCall]
def elaborateInvocation : TermElab := fun stx expected => do
  let stx ← PythonFullConstructors.prepareSource stx
  let raw:=stx.getArgs.back!
  let source ← match RawSourceText.normalize raw.getAtomVal with
    | .ok source=>pure source | .error message=>throwErrorAt raw "{message}"
  let parsed ← match Parser.runParserCategory (← getEnv) `term ("triton {\n"++source++"\n}") with
    | .ok parsed=>pure parsed | .error message=>throwErrorAt raw "{message}"
  let `(triton { $statements:tritonStmt* }) := parsed | throwError "expected a complete caller body"
  unless statements.any (fun statement=>containsAtomic statement.raw) do
    return ← CallSignature.elaborateInvocation stx expected
  let (formals,floating) ← pointerNames stx[1]
  let lowered ← liftMacroM <| lower formals (floating.map (fun name=>(name,name))) statements.toList
  let raw:=Syntax.atom raw.getHeadInfo ("{\n"++String.intercalate "\n" (lowered.map (PythonFullConstructors.render ·.raw))++"\n}")
  CallSignature.elaborateInvocation (stx.setArgs (stx.getArgs.set! (stx.getArgs.size-1) raw)) expected
end VeriTile.Triton.DSL.ImplicitAtomicOperands
